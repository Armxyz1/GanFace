import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
