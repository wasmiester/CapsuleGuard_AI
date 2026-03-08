import json
import base64
import cv2
import numpy as np
import torch
import os
from confluent_kafka import Consumer
from anomalib.deploy import TorchInferencer
import psycopg2

os.environ["TRUST_REMOTE_CODE"] = "1"

MODEL_PATH = "./results/exported_model/weights/torch/model.pt"
KAFKA_CONF = {'bootstrap.servers': 'localhost:9092', 'group.id': 'qc_group', 'auto.offset.reset': 'latest'}
DB_CONF = "dbname=capsule_inspection user=admin password=password123 host=localhost"

print("--- Loading PatchCore Brain ---")
inferencer = TorchInferencer(path=MODEL_PATH, device="cuda")

def log_to_db(status, score, image_path):
    try:
        conn = psycopg2.connect(DB_CONF)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO inspection_logs (status, image_path) VALUES (%s, %s)",
            (f"{status} (Score: {score:.2f})", image_path)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB Log Error: {e}")

def start_inspection():
    consumer = Consumer(KAFKA_CONF)
    consumer.subscribe(['raw_frames'])
    print("--- LIVE QC INSPECTION ACTIVE ---")

    while True:
        msg = consumer.poll(0.1)
        if msg is None: continue
        
        # Decode Frame
        data = json.loads(msg.value().decode('utf-8'))
        img_bytes = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        predictions = inferencer.predict(image=frame)
        
        # Score Anomoly
        score = predictions.pred_score.item()   # .item() turns the tensor into a float
        is_anomaly = bool(predictions.pred_label.item()) # ensures it's a standard True/False
        
        # Result Handeling
        status = "REJECT" if is_anomaly else "ACCEPT"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0) # Red = bad, Green = good

        # Visual Feedback for the logs
        cv2.putText(frame, f"{status} | SCORE: {score:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save frame to inspected folder
        save_path = f"data/inspected/frame_{data['timestamp']}.jpg"
        cv2.imwrite(save_path, frame)
        
        # Record the decision in Postgres
        log_to_db(status, score, save_path)
        print(f"[{data['timestamp']}] Decision: {status} | Score: {score:.4f}")

        if hasattr(predictions, 'anomaly_map') and predictions.anomaly_map is not None:
            # Pull from GPU to CPU and convert to numpy
            heatmap = predictions.anomaly_map.cpu().numpy().squeeze()
            
            # Normalize to 0-255 range
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_norm = heatmap_norm.astype(np.uint8)
            
            # Apply a 'Jet' colormap (Blue = Good, Red = Bad)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # Blend it with the original frame (50% transparency)
            heatmap_frame = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
        else:
            # Fallback if no map is available
            heatmap_frame = frame 

        # Save the visual evidence
        heatmap_path = f"data/inspected/heatmap_{data['timestamp']}.jpg"
        cv2.imwrite(heatmap_path, heatmap_frame)
        cv2.imwrite(f"data/inspected/heatmap_{data['timestamp']}.jpg", heatmap_frame) 

if __name__ == "__main__":
    os.makedirs("data/inspected", exist_ok=True)
    start_inspection()