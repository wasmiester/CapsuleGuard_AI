import torch
import os
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms
from anomalib.models import Patchcore

# 1. Bypass Security
os.environ["TRUST_REMOTE_CODE"] = "1"
torch.serialization.add_safe_globals([Patchcore])
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TEST_IMG = cfg["ai_settings"]["test_image"]

def test_brain():
    model_path = Path("./results/exported_model/weights/torch/model.pt").resolve()
    image_path = Path().resolve()

    # 2. Direct Load
    # The file *is* the model, so we just load it into the variable 'model'
    print("🧠 Loading full Patchcore object...")
    model = torch.load(model_path, map_location="cuda", weights_only=False)
    
    # If the load returns a dict with the model inside, grab it
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
        
    model.cuda().eval()

    # 3. Standard Pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Inference
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img_tensor)
        
        # Determine if output is a Batch object or a tuple
        if hasattr(output, 'pred_score'):
            raw_score = output.pred_score.item()
        elif isinstance(output, tuple):
            _, raw_score = output
            raw_score = raw_score.item()
        else:
            raw_score = output['pred_score'].item()

    print(f"\n🔍 Image: {image_path.name}")
    print(f"📊 Raw Anomaly Score: {raw_score:.4f}")
    
    # Using 25.0 as our temporary baseline
    threshold = 25.0 
    status = "PASS" if raw_score < threshold else "FAIL"
    print(f"🏁 Result: {status}")

if __name__ == "__main__":
    test_brain()