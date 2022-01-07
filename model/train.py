import cv2
import numpy as np
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class myOwnDataset(torch.utils.data.Dataset):
    '''Converts JSON file to a useable Coco dataset.'''
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # Path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # Open the input image
        img = Image.open(os.path.join(self.root, path))
        # Number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def get_transform():
    '''Transform data to Pytorch tensor using ToTensor.'''
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    '''Needed for batch training.'''
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    '''Loads a pretrained instance segmentation model (Faster R-CNN),
    get number of input features for the classifier,
    replace pretrained head with a fresh one to be trained'''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(model, data_loader, optimizer):
    '''Trains model.'''
    len_dataloader = len(data_loader)

    for epoch in range(config.num_epochs):
        print(f"Epochs Trained: {epoch}/{config.num_epochs}")
        model.train()
        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
        torch.save(model, "model/trained_detector.pth")
        print("Model saved.")


def predict_test(model_path, testimg_path):
    model = torch.load(model_path)
    model.eval() # Set to evaluation mode
    img = Image.open(testimg_path)
    transform = get_transform()
    img = transform(img).to(device)
    img = img.unsqueeze(0) # Add an extra first dimension, because the model predicts in batches
    
    with torch.no_grad():
        prediction = model(img)
        max_idx = np.argmax(prediction[0]["scores"].cpu().numpy())
        confidence = prediction[0]["scores"][max_idx].cpu().numpy()
        bbox_coords = prediction[0]["boxes"].cpu().numpy()[max_idx]
        bbox_coords = [int(x) for x in bbox_coords]
        image = cv2.imread(testimg_path)
        image = cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (0, 0, 255), 2)
        image = cv2.putText(image, f"{int(confidence*100)}%", (bbox_coords[0], bbox_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Output", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    import config # This is config.py
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    '''
    # Create custom dataset
    my_dataset = myOwnDataset(
        root=config.train_data_dir, annotation=config.train_coco, transforms=get_transform()
    )

    # Custom DataLoader
    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )

    # Faster R-CNN
    model = get_model_instance_segmentation(config.num_classes)

    # Make sure model training uses GPU (or CPU if you don't have a GPU)
    model.to(device)

    # Hyperparameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )

    train(model, data_loader, optimizer)
    '''
    predict_test("model/trained_detector.pth", "model/data/train/2022-01-06-16-31-08-077769.png")
