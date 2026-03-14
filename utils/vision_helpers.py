import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class VisionHelper:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.padding = 20 # Tighter padding for multiple capsules

    def get_all_capsules(self, frame):
        """Finds every pill in the frame and returns their locations and crops."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find all external shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        capsules = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: # Ignore tiny noise/dust
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Crop logic
            x_p, y_p = max(0, x-self.padding), max(0, y-self.padding)
            w_p, h_p = min(frame.shape[1]-x_p, w+self.padding*2), min(frame.shape[0]-y_p, h+self.padding*2)
            crop = frame[y_p:y_p+h_p, x_p:x_p+w_p]
            
            capsules.append({
                "crop": crop,
                "bbox": (x, y, w, h),
                "contour": cnt
            })
            
        return capsules

    def prepare_crop(self, crop):
        rgb_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        return self.transform(pil_img).unsqueeze(0).cuda()