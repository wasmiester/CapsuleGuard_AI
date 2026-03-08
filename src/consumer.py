import json
import base64
import os
import cv2
import numpy as np
import psycopg2
from confluent_kafka import Consumer

KAFKA_CONF = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'storage_group',
    'auto.offset.reset': 'earliest'
}

DB_CONF = "dbname=capsule_inspection user=admin password=password123 host=localhost"
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
    consumer = Consumer(KAFKA_CONF)
    consumer.subscribe(['raw_frames'])
    print("--- Consumer Active: Awaiting frames from Kafka ---")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Decode Payload
            data = json.loads(msg.value().decode('utf-8'))
            img_bytes = base64.b64decode(data['image'])
            
            # Convert to Image and Save
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            filename = f"frame_{int(data['timestamp'])}.jpg"
            filepath = os.path.join(DATA_DIR, filename)
            cv2.imwrite(filepath, frame)

            # Log to Postgres
            save_to_db(filepath)
            print(f"Saved: {filename} to Storage & Database")

    except KeyboardInterrupt:
        consumer.close()

if __name__ == "__main__":
    run_consumer()