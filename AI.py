import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. Define the Lane class with corner labeling
class Lane:
    def __init__(self, name, polygon_pts, color):
        self.name = name
        self.pts = np.array(polygon_pts, np.int32)
        self.color = color
        self.count = 0

    def reset_count(self):
        self.count = 0

    def contains_point(self, point):
        # Checks if vehicle tire point (cx, cy) is inside the polygon
        return cv2.pointPolygonTest(self.pts, point, False) >= 0

    def draw(self, frame, show_corners=True):
        # Draw main polygon outline
        cv2.polylines(frame, [self.pts], isClosed=True, color=self.color, thickness=2)

        # Draw and label each individual corner point
        if show_corners:
            for idx, pt in enumerate(self.pts):
                x, y = int(pt[0]), int(pt[1])
                
                # Red dot at the corner
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
                # Text label showing point index and coordinates: e.g. P0:(100,200)
                label = f"P{idx}:({x},{y})"
                cv2.putText(frame, label, (x + 6, y - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# 2. Load Model
print("Loading model...")
model = YOLO('rtdetr-l.pt')
VEHICLE_CLASSES = [2, 3, 5, 7] # Car, Motorcycle, Bus, Truck

# 3. Create 4 Lane instances
# Note: Update polygon coordinates for Lane C & D to match your video perspective
lanes = [
    Lane("Lane A", [[502, 202], [590, 110], [465, 0], [250, 0]], (255, 0, 0)),        # Blue
    Lane("Lane B", [[830, 230], [930, 320], [1250, 0], [1030, 0]], (0, 255, 255)),  # Yellow
    Lane("Lane C", [[750, 620], [833, 525], [1080, 720], [874, 719]], (0, 255, 0)),      # Green
    Lane("Lane D", [[455, 460], [518, 525], [300, 720], [125, 720]], (255, 0, 255))  # Magenta
]

# 4. State Machine Parameters
MIN_GREEN_TIME = 20.0
VEHICLE_THRESHOLD = 10
current_green_idx = 0  # Start with Lane A
green_start_time = time.time()

# 5. Open Video
video_path = r"C:\Users\shour\Downloads\3063475-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open '{video_path}'")
    exit()

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: [{x}, {y}]")
cv2.namedWindow("Smart AI Traffic Light System")
cv2.setMouseCallback("Smart AI Traffic Light System", click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
        continue

    # Resize full frame to 1280x720 window
    frame = cv2.resize(frame, (1280, 720))

    # Reset vehicle counts and draw polygons with labeled corner points
    for lane in lanes:
        lane.reset_count()
        lane.draw(frame, show_corners=True)

    # Run detection with imgsz=1088 (divisible by 32 stride)
    results = model(frame, conf=0.15, imgsz=1088, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), y2 # Bottom-center tire point
            
            # Draw tire point dot
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Check which lane contains this vehicle
            for lane in lanes:
                if lane.contains_point((cx, cy)):
                    lane.count += 1
                    break

    # Timer & Switch Logic (Highest Traffic Priority)
    elapsed_time = time.time() - green_start_time

    if elapsed_time >= MIN_GREEN_TIME:
        # Find waiting lane with the highest vehicle count
        waiting_lane_indices = [i for i in range(len(lanes)) if i != current_green_idx]
        highest_count_idx = max(waiting_lane_indices, key=lambda i: lanes[i].count)

        if lanes[highest_count_idx].count >= VEHICLE_THRESHOLD:
            current_green_idx = highest_count_idx
            green_start_time = time.time()
            elapsed_time = 0.0
            print(f"🚦 Switching light to {lanes[current_green_idx].name}!")

    
    # Display HUD dynamically for all lanes
    for idx, lane in enumerate(lanes):
        cv2.putText(frame, f"{lane.name} Count: {lane.count}", (30, 40 + idx * 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane.color, 2)

    y_offset = 40 + len(lanes) * 35
    cv2.putText(frame, f"Timer: {int(elapsed_time)}s / {int(MIN_GREEN_TIME)}s", (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"ACTIVE GREEN: {lanes[current_green_idx].name}", (30, y_offset + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    cv2.imshow("Smart AI Traffic Light System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
