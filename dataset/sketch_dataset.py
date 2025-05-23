from torch.utils.data import Dataset
import cv2
import os
from PIL import Image

class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, original_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.original_dir = original_dir
        self.image_files = sorted(os.listdir(sketch_dir))  # Ensure matching order
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    # def __getitem__(self, idx):
    #     sketch_path = os.path.join(self.sketch_dir, self.image_files[idx])
    #     original_path = os.path.join(self.original_dir, self.image_files[idx])

    #     sketch = Image.open(sketch_path).convert("RGB")
    #     original = Image.open(original_path).convert("RGB")

    #     if self.transform:
    #         sketch = self.transform(sketch)
    #         original = self.transform(original)

    #     return sketch, original  # input, target
    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.image_files[idx])
        original_path = os.path.join(self.original_dir, self.image_files[idx])

        sketch = cv2.imread(sketch_path)  # BGR format
        original = cv2.imread(original_path)

        # Convert BGR to RGB
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Apply the same transform to both images
        transformed = self.transform(image=sketch, mask=original)
        sketch = transformed["image"]
        original = transformed["mask"]


        return sketch, original  # input, target