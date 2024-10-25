import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import numpy as np
import kagglehub

# Custom dataset class
class ProcessedDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in sorted(os.listdir(root_dir)) if f.endswith('.jpg') or f.endswith('.png')]

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Dummy target for demonstration purposes
        target = {
            'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }

        return image, target

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a pre-trained Faster R-CNN model and modify it
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Download the dataset
path = kagglehub.dataset_download("trainingdatapro/license-plates-1-209-438-ocr-plates")
print("Path to dataset files:", path)

# Initialize dataset and dataloader
dataset = ProcessedDataset(root_dir=path, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model
model = get_model(num_classes=2)  # 1 class (e.g., license plate) + background
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {losses.item()}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Function to detect and blur sensitive areas
def detect_and_blur_sensitive_areas(image_path, output_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
        if score > 0.8 and label == 1:  # Assuming 'license plate' label corresponds to sensitive area
            x1, y1, x2, y2 = box.int().cpu().numpy()
            roi = img[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            img[y1:y2, x1:x2] = blurred_roi

    cv2.imwrite(output_path, img)

# Example usage for blurring
detect_and_blur_sensitive_areas('input.jpg', 'output.jpg', model)
