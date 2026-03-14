import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path

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

    print("---Building the Memory Bank---")
    
    engine.fit(model=model, datamodule=datamodule)
    engine.export(
        model=model,
        export_type="torch",
        export_root="./results/exported_model"
    )
    
    print("--- Success! Model exported to ./results/exported_model ---")

if __name__ == "__main__":
    train_brain()