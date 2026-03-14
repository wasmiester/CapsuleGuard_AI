import cv2
import os
import time
import sys
import yaml
from pathlib import Path

# Pathing setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vision_helpers import VisionHelper

# 1. LOAD CONFIG
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Navigate dataset pathing from YAML
# We combine 'root' and 'normal_dir' to save straight to your training folder
DATASET_ROOT = Path(cfg['dataset']['root'])
NORMAL_DIR = cfg['dataset']['normal_dir']
SAVE_PATH = DATASET_ROOT / NORMAL_DIR

if not SAVE_PATH.exists(): 
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Hardware Settings from YAML
CAM_INDEX = cfg['camera_settings']['device_index']
FOCUS_VAL = cfg['camera_settings']['default_focus']

# 2. INITIALIZE
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
vh = VisionHelper()

# Apply Hardware Locks (using C920 optimization)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VAL)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, 0)

time.sleep(2)

print(f"--- DATA CAPTURE ACTIVE ---")
print(f"Saving to: {SAVE_PATH}")
print("Press 'S' to Save | 'F' for Camera Settings | 'Q' to Quit")

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Use the class-based VisionHelper from your updated file
    capsules = vh.get_all_capsules(frame)
    display_raw = frame.copy()
    active_crop = None
    can_save = False

    H, W = frame.shape[:2]

    # Process detections
    for cap_data in capsules:
        x, y, w, h = cap_data["bbox"]
        
        # Check if centered (Safe Zone)
        is_touching_edge = (x <= 15 or y <= 15 or (x + w) >= W - 15 or (y + h) >= H - 15)
        
        if is_touching_edge:
            color = (0, 0, 255) # RED
            cv2.putText(display_raw, "MOVE MINT TO CENTER", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            color = (0, 255, 0) # GREEN
            can_save = True
            active_crop = cap_data["crop"]

        cv2.drawContours(display_raw, [cap_data["contour"]], -1, color, 2)

    # UI Overlay
    cv2.putText(display_raw, f"SAMPLES IN SESSION: {count}", (10, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("1. RAW FEED", display_raw)
    if active_crop is not None:
        cv2.imshow("2. CROP PREVIEW", active_crop)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s') and can_save:
        # Day 7 Hardening: Unique filename to prevent overwrites
        timestamp = int(time.time())
        img_name = f"good_{timestamp}_{count}.jpg"
        final_path = SAVE_PATH / img_name
        
        if frame is not None:
            cv2.imwrite(str(final_path), frame)
            print(f"✅ Saved: {img_name}")
            count += 1
        else:
            print("Error: Frame is empty, cannot save.")
    
    elif key == ord('f'):
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()