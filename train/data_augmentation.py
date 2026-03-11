import cv2
import os
import numpy as np

INPUT_DIR = "dataset/my_capsules/raw_photos"
OUTPUT_DIR = "dataset/my_capsules/train/good"

def text_safe_augment():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get all images in raw folder
    raw_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in raw_files:
        img = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img is None: continue
        
        h, w = img.shape[:2]
        base_name = os.path.splitext(filename)[0]
        
        # 1. Rotations (Every 45 degrees to cover all belt orientations)
        for angle in range(0, 360, 45):
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h))
            
            # 2. Zoom Levels (Simulating camera distance/zoom)
            for scale in [0.75, 1.0, 1.25]:
                new_h, new_w = int(h * scale), int(w * scale)
                
                if scale < 1.0:
                    # Zoom Out: Resize and Pad
                    resized = cv2.resize(rotated, (new_w, new_h))
                    pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                    final = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, 
                                               pad_w, w-new_w-pad_w, 
                                               cv2.BORDER_CONSTANT, value=[0,0,0])
                else:
                    # Zoom In: Resize and Crop center
                    resized = cv2.resize(rotated, (new_w, new_h))
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    final = resized[start_h:start_h+h, start_w:start_w+w]
                
                # Save the result
                save_path = f"{OUTPUT_DIR}/{base_name}_rot{angle}_scale{scale}.jpg"
                cv2.imwrite(save_path, final)

if __name__ == "__main__":
    text_safe_augment()
    print(f"Augmentation complete! Check {OUTPUT_DIR}")