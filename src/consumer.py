import json
import base64
import os
import cv2
import numpy as np
import psycopg2
import yaml
from confluent_kafka import Consumer

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DB_CONF = cfg['database']['connection_string']
RAW_TOPIC = cfg['kafka_settings']['raw_topic']
RAW_TOPIC = cfg['kafka_settings']['raw_topic']
DATA_DIR = "data/raw_frames"
os.makedirs(DATA_DIR, exist_ok=True)

def save_to_db(image_path):
    conn = psycopg2.connect(DB_CONF)
    cur = conn.cursor()
    cur.execute("INSERT INTO inspection_logs (image_path) VALUES (%s)", (image_path,))
    conn.commit()
    cur.close()
    conn.close()

def run_consumer():
    conf = {'bootstrap.servers': cfg['kafka_settings']['bootstrap_servers'], 
                'group.id': 'storage_group'}
    consumer = Consumer(conf)
    consumer.subscribe([RAW_TOPIC])
    
    print("--- Consumer Active: Awaiting frames from Kafka ---")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Decode Payload
            data = json.loads(msg.value().decode('utf-8')) # type: ignore
            img_bytes = base64.b64decode(data['image'])
            
            # Convert to Image and Save
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            filename = f"frame_{int(data['timestamp'])}.jpg"
            filepath = os.path.join(DATA_DIR, filename)
            if frame is not None:
                cv2.imwrite(str(filepath), frame)
            else:
                print("Error: Frame is empty, cannot save.")

            # Log to Postgres
            save_to_db(filepath)
            print(f"Saved: {filename} to Storage & Database")

    except KeyboardInterrupt:
        consumer.close()

if __name__ == "__main__":
    run_consumer()