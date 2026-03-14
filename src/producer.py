import cv2
import base64
import yaml
from confluent_kafka import Producer

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
DEVICE_INDEX = cfg['camera_settings']['device_index']
conf = {'bootstrap.servers': cfg['kafka_settings']['bootstrap_servers']}
DEVICE_INDEX = cfg['camera_settings']['device_index']
FOCUS_VAL = cfg['camera_settings']['default_focus']

producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

def run_virtual_sensor():
    cap = cv2.VideoCapture(DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VAL)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    
    prev_gray = None
    stable_frames = 0
    
    print("--- CapsuleGuard AI: Virtual Sensor Active ---")
    print("Place a capsule under the camera to trigger inspection.")

KAFKA_SERVER = 'localhost:9092'
TARGET_WIDTH = 640 
TARGET_HEIGHT = 480

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    # Optimization: Allow larger batches and gzip compression
    batch_size=32768,
    linger_ms=10,
    compression_type='gzip'
)

cap = cv2.VideoCapture(1)

print("--- PRODUCER ACTIVE: Optimized for 30FPS ---")

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

cap.release()