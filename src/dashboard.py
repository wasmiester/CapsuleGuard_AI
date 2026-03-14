import streamlit as st
import json
import base64
import numpy as np
from confluent_kafka import Consumer

st.set_page_config(page_title="CapsuleGuard AI", layout="wide")

KAFKA_CONF = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'dashboard_group',
    'auto.offset.reset': 'latest'
}

st.title("CapsuleGuard Live Inspector")
placeholder = st.empty()

consumer = Consumer(KAFKA_CONF)
consumer.subscribe(['processed_frames'])

while True:
    msg = consumer.poll(1.0)
    if msg is None: continue
    
    data = json.loads(msg.value().decode('utf-8')) # type: ignore
    
    with placeholder.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(base64.b64decode(data['image']), use_container_width=True)
            
        with col2:
            st.metric("System Status", data['status'])
            st.write("### Detection Logs")
            for i, res in enumerate(data['results']):
                color = "green" if res['status'] == "PASS" else "red"
                st.markdown(f"**Pill {i}:** :{color}[{res['status']}] (Score: {res['score']:.2f})")