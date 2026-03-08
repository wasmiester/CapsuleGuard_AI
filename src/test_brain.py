import os
import torch
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from pathlib import Path

os.environ["TRUST_REMOTE_CODE"] = "1"
torch.set_float32_matmul_precision('medium')

def test_brain():
    print("--- Day 3: Modern Inference Test ---")
    
    model = Patchcore(backbone="wide_resnet50_2", pre_trained=True)
    
    CHECKPOINT_PATH = list(Path("./results").rglob("*.ckpt"))[0]
    
    engine = Engine()

    datamodule = MVTecAD(root=Path("./data"), category="capsule")
    datamodule.prepare_data = lambda: None
    
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    predictions = engine.predict(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=str(CHECKPOINT_PATH)
    )

    print("--- Test Complete! ---")
    print("Check the './results' folder for a new 'predictions' directory.")
    print("You will see side-by-side comparisons of Good vs. Defective capsules.")

if __name__ == "__main__":
    test_brain()