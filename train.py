"""
Main file for training Yolo model on Pascal VOC dataset

"""
import config
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from Yolov1 import Yolov1
from Yolov3 import Yolov3
from dataset import YOLODataset
from utils import (
    non_max_suppression,
    intersection_over_union,
    get_bboxes,
    plot_image,
    mean_average_precision,
    cellboxes_to_boxes,
    cells_to_bboxes,
    save_checkpoint,
    load_checkpoint,
    get_evaluation_bboxes,
    check_class_accuracy,
    V1_get_loaders,
    V3_get_loaders,
    plot_couple_examples,
)
from loss import YoloV1Loss
from loss import YoloV3Loss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

seed = 123
torch.manual_seed(seed)

def train_fn(train_loader, model, optimizer, loss_fn, scaler=None, scaled_anchors=None):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        with torch.cuda.amp.autocast():
            if config.MODEL_NAME == 'Yolov1':
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                out = model(x)
                loss = loss_fn(out, y)
            elif config.MODEL_NAME == 'Yolov3':
                x = x.to(config.DEVICE)
                out = model(x)
                y0, y1, y2 = (
                    y[0].to(config.DEVICE),
                    y[1].to(config.DEVICE),
                    y[2].to(config.DEVICE),)
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2]))

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            if config.MODEL_NAME == 'Yolov1':
                loop.set_postfix(loss=loss.item())
            elif config.MODEL_NAME == 'Yolov3':
                loop.set_postfix(loss=mean_loss)

    print(f"Mean loss was {sum(losses)/len(losses)}")


def main():

    if config.MODEL_NAME == 'Yolov1':
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(config.DEVICE)
        loss_fn = YoloV1Loss()
        scaler = torch.cuda.amp.GradScaler()
        train_loader, test_loader = V1_get_loaders(
            train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        )

    elif config.MODEL_NAME == 'Yolov3':
        model = Yolov3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        loss_fn = YoloV3Loss()
        scaler = torch.cuda.amp.GradScaler()
        scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
        train_loader, test_loader, train_eval_loader = V3_get_loaders(
            train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # if config.LOAD_MODEL:
    #     load_checkpoint(torch.load(config.LOAD_MODEL_FILE), model, optimizer)


    for epoch in range(config.NUM_EPOCHS):

        if config.MODEL_NAME == 'Yolov1':
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec}")

            train_fn(train_loader, model, optimizer, loss_fn, scaler)


        elif config.MODEL_NAME == 'Yolov3':

            train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

            if epoch > 0 and epoch % 3 == 0:
                check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    test_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=config.CONF_THRESHOLD,
                )
                mapval = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=config.NUM_CLASSES,
                )
                print(f"MAP: {mapval.item()}")
                model.train()


if __name__ == "__main__":
    main()
