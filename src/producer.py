import cv2
import time
import json
import base64
from confluent_kafka import Producer

conf = {'bootstrap.servers': "localhost:9092"}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

def run_virtual_sensor():
    cap = cv2.VideoCapture(0) # to change camera change number to 1 or 2
    prev_gray = None
    stable_frames = 0
    
    print("--- CapsuleGuard AI: Virtual Sensor Active ---")
    print("Place a capsule under the camera to trigger inspection.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Pre-process for motion stability
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        # Check for Stillness
        delta = cv2.absdiff(prev_gray, gray)
        motion_level = delta.sum()
        if motion_level < 5000000: 
            stable_frames += 1
        else:
            stable_frames = 0

        # Trigger capture after 10 frames of stillness
        if stable_frames == 10:
            print(">> Target Settled. Sending 1080p frame to Kafka...")
            send_to_kafka(frame)
            stable_frames = -60 # Cooldown so it doesn't spam the same pill

        prev_gray = gray
        cv2.imshow("Factory Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def send_to_kafka(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "timestamp": time.time(),
        "image": img_b64,
        "metadata": {"sensor_id": "line_1_cam"}
    }
    
    producer.produce('raw_frames', json.dumps(payload), callback=delivery_report)
    producer.flush()

if __name__ == "__main__":
    run_virtual_sensor()