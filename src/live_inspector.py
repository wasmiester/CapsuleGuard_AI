import cv2
import numpy as np
import base64
import json
from confluent_kafka import Consumer
import torch
from torchvision import transforms
import time

MODEL_PATH = "results/exported_model/weights/torch/model.pt"
KAFKA_SERVER = "localhost:9092"

THRESHOLD = 0.85      
MIN_AREA = 1500        
MAX_AREA = 60000       
MIN_CIRCULARITY = 0.4  

BUFFER_SIZE = 3  
SHOW_HEATMAP = False  

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("--- Loading Model ---")
model = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)
model.to("cuda")
model.eval()


kafka_conf = {
    'bootstrap.servers': KAFKA_SERVER,
    'group.id': f'inspector-group-{time.time()}', 
    'auto.offset.reset': 'latest',
    'enable.auto.commit': False,
    'fetch.min.bytes': 1,
    'socket.receive.buffer.bytes': 262144
}
consumer = Consumer(kafka_conf)
consumer.subscribe(['raw_frames'])

def base64_to_cv2(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def start_inspection():
    print("--- LIVE QC INSPECTION ACTIVE ---")
    cv2.namedWindow("AI Quality Control", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            msg = consumer.poll(0.1)
            while True:
                next_msg = consumer.poll(0)
                if next_msg is None: break
                msg = next_msg

            if msg is None or msg.error(): continue
            
            data = json.loads(msg.value().decode('utf-8'))
            frame = base64_to_cv2(data['image'])
            display_frame = frame.copy()
            
            # 1. Detection Phase
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11,11), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            tablet_count = 0
            frame_scores = []

            # 2. Inspection Phase
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if MIN_AREA < area < MAX_AREA and circularity > MIN_CIRCULARITY:
                    tablet_count += 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Crop and Analyze
                    pad_w = int(w * 0.2) 
                    pad_h = int(h * 0.2)

                    y1, y2 = max(0, y-pad_h), min(frame.shape[0], y+h+pad_h)
                    x1, x2 = max(0, x-pad_w), min(frame.shape[1], x+w+pad_w)
                    capsule_crop = frame[y1:y2, x1:x2]

                    crop_rgb = cv2.cvtColor(capsule_crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(crop_rgb).unsqueeze(0).to("cuda")
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    score = float(output.pred_score.cpu().item())
                    frame_scores.append(score)

                    # Draw per-capsule boxes
                    box_color = (0, 0, 255) if score > THRESHOLD else (0, 255, 0)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(display_frame, f"SCR: {score:.2f}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 3. Status Logic
            if tablet_count == 0:
                status_label = "IDLE - NO OBJECT"
                main_color = (0, 255, 255)
                avg_score = 0.0
            else:
                # REJECT if ANY capsule in the frame is over threshold
                status_label = "REJECT" if any(s > THRESHOLD for s in frame_scores) else "ACCEPT"
                main_color = (0, 0, 255) if status_label == "REJECT" else (0, 255, 0)
                avg_score = np.mean(frame_scores)

            # 4. HUD RENDERING
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

            cv2.putText(display_frame, f"STATUS: {status_label}", (20, 45), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, main_color, 2)
            cv2.putText(display_frame, f"CAPSULES DETECTED: {tablet_count}", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, f"AVG ANOMALY SCORE: {avg_score:.3f}", (20, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("AI Quality Control", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        consumer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_inspection()