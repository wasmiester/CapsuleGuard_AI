from anomalib.models import get_model
import torch

model = get_model("patchcore")


checkpoint = torch.load("results/exported_model/weights/torch/model.pt", map_location="cpu")

if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
elif hasattr(checkpoint, "state_dict"):
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

torch.save(state_dict, "results/patchcore_clean_weights.pt")
print("--- CLEAN WEIGHTS SAVED ---")