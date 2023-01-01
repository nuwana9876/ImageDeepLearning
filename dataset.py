
import config
import numpy as np
import torch
import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors=None, image_size=448, S=[13, 26, 52], B=2 , C=20, transform=None,):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        # Yolov3 Anchor setting
        if anchors:
            self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
            self.num_anchors = self.anchors.shape[0]
            self.num_anchors_per_scale = self.num_anchors // 3
            self.ignore_iou_thresh = 0.5
        else:
            self.anchors = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        if self.anchors:
            # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
            targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]      
            for box in bboxes:
                iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
                anchor_indices = iou_anchors.argsort(descending=True, dim=0)
                x, y, width, height, class_label = box
                has_anchor = [False] * 3  # each scale should have one anchor
                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // self.num_anchors_per_scale
                    anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                    S = self.S[scale_idx]
                    i, j = int(S * y), int(S * x)  # which cell
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                        width_cell, height_cell = (
                            width * S,
                            height * S,
                        )  # can be greater than 1 since it's relative to cell
                        box_coordinates = torch.tensor(
                            [x_cell, y_cell, width_cell, height_cell]
                        )
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True

                    elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

            return image, tuple(targets)

        else:
            label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
            for box in bboxes:
                x, y, width, height, class_label = box
                class_label = int(class_label)

                # i,j represents the cell row and cell column
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i

                """
                Calculating the width and height of cell of bounding box,
                relative to the cell is done by the following, with
                width as the example:
                
                width_pixels = (width*self.image_width)
                cell_pixels = (self.image_width)
                
                Then to find the width relative to the cell is simply:
                width_pixels/cell_pixels, simplification leads to the
                formulas below.
                """
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                # If no object already found for specific cell i,j
                # Note: This means we restrict to ONE object
                # per cell!
                if label_matrix[i, j, 20] == 0:
                    # Set that there exists an object
                    label_matrix[i, j, 20] = 1

                    # Box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    label_matrix[i, j, 21:25] = box_coordinates

                    # Set one hot encoding for class_label
                    label_matrix[i, j, class_label] = 1

            return image, label_matrix