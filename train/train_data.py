import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path

# Performance tip for your RTX 4070
torch.set_float32_matmul_precision('medium')

def train_brain():
    print("--- Initializing AI Brain (PatchCore) ---")

    datamodule = MVTecAD(
        root=Path("./data"), 
        category="capsule",
        train_batch_size=32,
        eval_batch_size=32
    )

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
    
    full_save_path = Path("./results/exported_model/weights/torch/model.pt")
    full_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model, full_save_path)

    print(f"--- SUCCESS: Full object with Memory Bank saved to {full_save_path} ---")

if __name__ == "__main__":
    train_brain()