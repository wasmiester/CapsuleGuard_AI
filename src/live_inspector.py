import cv2
import torch
import os
from pathlib import Path
from anomalib.models import Patchcore
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vision_helpers import VisionHelper

os.environ["TRUST_REMOTE_CODE"] = "1"
torch.serialization.add_safe_globals([Patchcore])

class CapsuleInspector:
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        # Pathing: Looks for results folder in the project root
        model_path = Path(__file__).parent.parent / "results/exported_model/weights/torch/model.pt"

        # Load the model
        checkpoint = torch.load(model_path, map_location="cuda", weights_only=False)

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                self.model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                self.model = checkpoint["state_dict"] 
            else:
                self.model = list(checkpoint.values())[0]
        else:
            self.model = checkpoint

        if hasattr(self.model, "cuda"):
            self.model = self.model.cuda()
            self.model.eval()
        else:
            print("Error: Extracted object is not a Torch model.")
            print(f"Type found: {type(self.model)}")

    def detect(self):
        self.vh = VisionHelper()
        self.threshold = self.cfg['ai_settings']['threshold']
        self.focus_val = self.cfg['camera_settings']['default_focus']
        self.device_index = self.cfg['camera_settings']['device_index']
        
        print("Loading Patchcore Brain...")
        cap = cv2.VideoCapture(self.device_index)

        # Setup Camera Properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Lock Focus: Disable Auto (0) and set Manual value
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_FOCUS, self.focus_val)

        print(f"Inspection Active.")
        print(f"Controls: [W/S] Adjust Focus | [Q] Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detection: Get all capsules currently in view
            capsules = self.vh.get_all_capsules(frame)
            current_count = len(capsules)

            # 2. Inference: Process each detected capsule
            for cap_data in capsules:
                # Prepare the specific crop for the model
                img_tensor = self.vh.prepare_crop(cap_data["crop"])

                with torch.no_grad():
                    output = self.model(img_tensor)
                    score = output.pred_score.item() if hasattr(output, 'pred_score') else output[1].item()

                # Determine Pass/Reject
                status = "PASS" if score < self.threshold else "REJECT"
                color = (0, 255, 0) if status == "PASS" else (0, 0, 255)

                # Draw per-capsule box and score
                x, y, w, h = cap_data["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{status} {score:.1f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. UI: Global Information HUD
            cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"ON SCREEN: {current_count}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FOCUS: {self.focus_val}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 4. Display and Controls
            cv2.imshow('Centrum Quality Control - Live', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.focus_val = min(255, self.focus_val + 5)
                cap.set(cv2.CAP_PROP_FOCUS, self.focus_val)
                print(f"Focus increased to: {self.focus_val}")
            elif key == ord('s'):
                self.focus_val = max(0, self.focus_val - 5)
                cap.set(cv2.CAP_PROP_FOCUS, self.focus_val)
                print(f"Focus decreased to: {self.focus_val}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inspector = CapsuleInspector()