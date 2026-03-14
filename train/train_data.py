import os
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from anomalib.models import Patchcore
from anomalib.engine import Engine
from dotmap import DotMap

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# In data_augmentation.py
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Dynamically resolve paths
INPUT_DIR = Path(cfg['dataset']['root']) / "raw_photos"
OUTPUT_DIR = Path(cfg['dataset']['root']) / cfg['dataset']['normal_dir']

class SimpleCentrumDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.images = list(self.root_dir.glob("*.jpg"))[::3] 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        return DotMap({
            "image": self.transform(image),
            "label": 0,
            "gt_mask": torch.zeros((1, 224, 224))
        })

    def __len__(self):
        return len(self.images)

def train():
    torch.set_float32_matmul_precision('high')
    
    dataset = SimpleCentrumDataset(INPUT_DIR)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Optimized Loader: Training on {len(dataset)} representative images.")

    model = Patchcore(
        backbone="resnet18", 
        coreset_sampling_ratio=0.01 
    )

    engine = Engine(devices=1, accelerator="gpu", default_root_dir="./results")

    print("Training ResNet-18 Memory Bank on RTX 4070...")
    engine.fit(model=model, train_dataloaders=train_loader)

    print("Exporting model...")
    try:
        engine.export(
            model=model,
            export_type="torch", 
            export_root=Path(OUTPUT_DIR).resolve()
        )
    except TypeError:
        engine.export(
            model=model,
            export_mode="torch",  # type: ignore
            export_root=Path("./results/exported_model").resolve()
        )
    
    print("✅ SUCCESS! Check ./results/exported_model/weights/torch for your .pt or .torchscript file.")

if __name__ == "__main__":
    train()