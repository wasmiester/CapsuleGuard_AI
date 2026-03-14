import cv2
import numpy as np
import os

INPUT_DIR = "dataset/my_capsules/raw_photos"
OUTPUT_DIR = "dataset/my_capsules/train/good"

def augment_centrum_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    raw_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"Starting Multi-Brightness Augmentation for {len(raw_files)} images...")

    # Parameters
    angles = np.arange(0, 360, 22.5)  # 16 rotations
    scales = [0.9, 1.0, 1.1]          # 3 scales
    
    # --- MULTI-BRIGHTNESS LEVELS ---
    # 0.5 = 50% darker | 1.0 = Normal | 1.5 = 50% brighter
    brightness_levels = [0.5, 0.75, 1.0, 1.25, 1.5]

    total_count = 0

    for filename in raw_files:
        img = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img is None: continue
        
        h, w = img.shape[:2]
        base_name = os.path.splitext(filename)[0]

        for angle in angles:
            # 1. Rotate
            matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=(0,0,0))
            
            for scale in scales:
                # 2. Resize
                new_h, new_w = int(h * scale), int(w * scale)
                res = cv2.resize(rotated, (new_w, new_h))
                final_res = cv2.resize(res, (224, 224))
                
                # --- APPLY MULTIPLE BRIGHTNESS VARIATIONS ---
                for b_val in brightness_levels:
                    # alpha=b_val multiplies pixel values
                    # beta=0 means we don't add a flat constant (keeps blacks black)
                    final_img = cv2.convertScaleAbs(final_res, alpha=b_val, beta=0)
                    
                    save_name = f"{base_name}_rot{int(angle)}_s{scale}_b{b_val}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), final_img)
                    total_count += 1

    print(f"--- Augmentation Complete ---")
    print(f"Total Images Generated: {total_count}")
    print(f"Each original photo was turned into {16 * 3 * 5} variations.")

if __name__ == "__main__":
    augment_centrum_dataset()