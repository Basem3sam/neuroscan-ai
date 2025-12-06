# dataset.py
import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
from config import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED
import cv2

# Histogram equalization transform
class HistEqualize:
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:  # Convert RGB to grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        eq_img = cv2.equalizeHist(img_np)
        return Image.fromarray(eq_img)

# Compose transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    HistEqualize(),
    transforms.ToTensor()
])

def get_dataloaders():
    ct_path = os.path.join(DATA_ROOT, "Brain Tumor CT scan Images")
    mri_path = os.path.join(DATA_ROOT, "Brain Tumor MRI images")

    ct = datasets.ImageFolder(ct_path, transform=transform)
    mri = datasets.ImageFolder(mri_path, transform=transform)

    if ct.class_to_idx != mri.class_to_idx:
        raise ValueError("CT & MRI classes mismatch!")

    dataset = ConcatDataset([ct, mri])
    targets = [label for _, label in ct.samples] + [label for _, label in mri.samples]

    train_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=RANDOM_SEED
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, ct.classes
