import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
import os

model = YOLO('D:\\Source Codes\\TA Accident\\runs_atcs\\detect\\train\\weights\\best.pt')

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.9)

def CarConditionStatus(cls):
    return 'damaged' if cls == 0 else 'normal'

def output_dir(cropped_dir):
    os.makedirs(cropped_dir, exist_ok=True)

def process_frame(frame, threshold_conf, valid_classes):
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes:
            for detection in result.boxes:
                conf = detection.conf.item()
                cls = int(detection.cls.item())
                if conf >= threshold_conf and cls in valid_classes:
                    x1, y1, x2, y2 = detection.xyxy[0].tolist()
                    left, top = float(x1), float(y1)
                    width, height = float(x2 - x1), float(y2 - y1)
                    detections.append(([left, top, width, height], float(conf), str(cls)))
    return detections

def crop_and_save_image(frame, track, cropped_dir, frame_count):
    track_id = track.track_id
    ltrb = track.to_ltrb()
    x1, y1, x2, y2 = map(int, ltrb)

    # Crop image
    crop_left = max(0, x1 - 10)
    crop_top = max(0, y1 - 10)
    crop_right = min(frame.shape[1], x2 + 10)
    crop_bottom = min(frame.shape[0], y2 + 10)

    # crop 300x300
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width < 300:
        diff = 300 - crop_width
        crop_left = max(0, crop_left - diff // 2)
        crop_right = min(frame.shape[1], crop_right + diff // 2)
    if crop_height < 300:
        diff = 300 - crop_height
        crop_top = max(0, crop_top - diff // 2)
        crop_bottom = min(frame.shape[0], crop_bottom + diff // 2)

    cropped_image = frame[crop_top:crop_bottom, crop_left:crop_right]
    cv2.imwrite(os.path.join(cropped_dir, f'cropped_frame_{frame_count}_track_{track_id}.jpg'), cropped_image)

def plot_frame(ax, frame, frame_count, tracks, anomaly_vehicles):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # bounding boxes
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        condition = anomaly_vehicles.get(track.track_id, 'unknown')
        color = (255, 0, 0) if condition == 'damaged' else (0, 255, 0)
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_rgb, f'ID: {track.track_id} - {condition}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax.imshow(frame_rgb)
    ax.set_title(f'Frame {frame_count}')
    ax.axis('off')
    plt.draw()
    plt.pause(0.1)
    ax.clear()

def plot_detections_per_frame(detections_per_frame):
    """Plot the number of vehicle detections per frame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(detections_per_frame, marker='o', linestyle='-', color='b')
    ax.set_title('Number of Vehicle Detections Per Frame')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Number of Detections')
    ax.grid(True)
    plt.show()

def process_video(video_path, threshold_conf, frame_skip=1, valid_classes=[0, 1]):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomaly_vehicles = {}
    detections_per_frame = []
    cropped_dir = 'images/cropped'
    output_dir(cropped_dir)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        detections = process_frame(frame, threshold_conf, valid_classes)
        print(f"Frame {frame_count} detections: {detections}")

        if not all(isinstance(det, tuple) and len(det) == 3 for det in detections):
            print(f"Invalid detection format in frame {frame_count}: {detections}")
            continue

        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Error updating tracks: {e}")
            continue

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls = int(track.det_class) if hasattr(track, 'det_class') else -1

            if track_id in anomaly_vehicles and anomaly_vehicles[track_id] == 'damaged':
                condition = 'damaged'
            else:
                condition = CarConditionStatus(cls)
                anomaly_vehicles[track_id] = condition

            if condition == 'damaged':
                crop_and_save_image(frame, track, cropped_dir, frame_count)

            print(f"Track ID: {track_id}, Class: {cls}, Condition: {condition}")

        detections_per_frame.append(len(detections))
        plot_frame(ax, frame, frame_count, tracks, anomaly_vehicles)
        frame_count += 1

    cap.release()
    plt.ioff()
    plot_detections_per_frame(detections_per_frame)
    print("Anomaly Vehicles:", anomaly_vehicles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video with vehicle detection and tracking.')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--threshold_conf', type=float, default=0.4, help='Confidence threshold for detections.')
    parser.add_argument('--frame_skip', type=int, default=1, help='Number of frames to skip between processing.')
    parser.add_argument('--valid_classes', type=int, nargs='+', default=[0, 1], help='List of valid detection classes.')
    args = parser.parse_args()
    process_video(args.video, args.threshold_conf, args.frame_skip, args.valid_classes)
