import cv2
import base64
import json
import time
from kafka import KafkaProducer

KAFKA_SERVER = 'localhost:9092'
TARGET_WIDTH = 640 
TARGET_HEIGHT = 480

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    batch_size=32768,
    linger_ms=10,
    compression_type='gzip'
)

cap = cv2.VideoCapture(1)

print("--- PRODUCER ACTIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Resize
    small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # 2. Compress 
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, buffer = cv2.imencode('.jpg', small_frame, encode_param)

    b64_frame = base64.b64encode(buffer).decode('utf-8')

    payload = {
        'timestamp': time.time(),
        'image': b64_frame
    }

    producer.send('raw_frames', value=payload)

    # 3. sleep to prevent CPU pegged at 100%
    time.sleep(0.01)