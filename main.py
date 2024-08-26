from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import cv2
import numpy as np
import datetime
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

deepsort = DeepSort()

cap = cv2.VideoCapture(0)

image_saved = False

current_directory = os.path.dirname(os.path.abspath(__file__))

photos_directory = os.path.join(current_directory, 'photo')

if not os.path.exists(photos_directory):
    os.makedirs(photos_directory)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    detections = results.xyxy[0].cpu().numpy()

    print("Shape of detections:", detections.shape)
    if detections.shape[0] > 0:
        print("Sample detection:", detections[0])

    formatted_detections = []
    for detection in detections:
        print("Detection values:", detection)
        x1, y1, x2, y2, conf = detection[:5]
        formatted_detections.append([x1, y1, x2, y2, conf])
    formatted_detections = np.array(formatted_detections)

    try:
        tracks = deepsort.update_tracks(formatted_detections, frame)
    except Exception as e:
        print(f"Error in updating tracks: {e}")
        continue

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if conf > 0.5 and not image_saved:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(photos_directory, f'detected_object_{timestamp}.jpg')
            if cv2.imwrite(filename, frame):
                print(f"Saved image: {filename}")
                image_saved = True
            else:
                print(f"Failed to save image: {filename}")

    cv2.imshow('Object Detection with Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
