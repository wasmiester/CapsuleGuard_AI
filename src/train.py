import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path

# Performance tip for your RTX 4070
torch.set_float32_matmul_precision('medium')

def train_brain():
    print("--- Initializing AI Brain (PatchCore) ---")

    # The 'root' should be the folder CONTAINING the 'capsule' directory, 
    # OR point directly to it if configured correctly.
    # If your images are in ./data/capsule/train/good, use root="./data"
    datamodule = MVTecAD(
        root=Path("./data"), 
        category="capsule",
        train_batch_size=32,
        eval_batch_size=32
    )

    # Manual override: tell Anomalib the data is already there so it skips download
    datamodule.prepare_data = lambda: None 

    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True
    )

    engine = Engine(
        max_epochs=1,
        default_root_dir="./results"
    )

    print("--- Building the Memory Bank... Utilizing RTX 4070 ---")
    
    engine.fit(model=model, datamodule=datamodule)

    engine.export(
        model=model,
        export_type="torch",
        export_root="./results/exported_model"
    )

    print("--- Success! Model exported to ./results/exported_model ---")

if __name__ == "__main__":
    train_brain()