import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Forces PyTorch to report the exact error line

import cv2
import streamlink
from ultralytics import YOLO

print("Loading RT-DETR model...")
# Use the RT-DETR Large model
model = YOLO('rtdetr-l.pt')


VEHICLE_CLASSES = [2, 3, 5, 7]

youtube_url = "https://www.youtube.com/watch?v=cDuVtH0CZk"

cap = None

try:
    print("Fetching stream links...")
    streams = streamlink.streams(youtube_url)
    
    if not streams:
        raise ValueError("No streams found. The YouTube live stream might be offline or invalid.")
    
    
    stream_url = streams['720p'].to_url() if '720p' in streams else streams['best'].to_url()
    cap = cv2.VideoCapture(stream_url)
    print("Connected to live YouTube stream!")

except Exception as e:
    print(f"Streamlink Error: {e}")
    print("Falling back to local camera/video file...")
    # Fallback to local MP4 video or webcam (0)
    cap = cv2.VideoCapture("indian_traffic.mp4")

while cap and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of stream or frame unavailable.")
        break

    # 2. Run YOLO detection on the frame (verbose=False silences console logging)
    results = model(frame, verbose=False)

    # 3. Count detected vehicles
    vehicle_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            vehicle_count += 1

    # 4. Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # 5. Overlay vehicle count HUD text on top left
    cv2.putText(
        annotated_frame, 
        f"Vehicle Count: {vehicle_count}", 
        (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.2, 
        (0, 255, 0), 
        3
    )

    # 6. Display the annotated output
    cv2.imshow("AI Traffic Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()