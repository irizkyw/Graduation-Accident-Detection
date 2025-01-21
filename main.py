import argparse
import os
import sys
from src.Configs import *
from src.Detection import *
from src.Processors import *

def path_videos(directory):
    video_extensions = ('.mp4', '.avi', '.mov')
    videos = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                videos.append(os.path.join(root, file))
    return videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video or folder with vehicle detection.')
    parser.add_argument('--video', type=str, help='Path to the video file or directory.')
    parser.add_argument('--all_videos', action='store_true', help='Process all videos in the specified folder and subfolders.')
    parser.add_argument('--threshold_conf', type=float, default=0.7, help='Confidence threshold for detections.')
    parser.add_argument('--frame_skip', type=int, default=1, help='Number of frames to skip between processing.')
    parser.add_argument('--valid_classes', type=int, nargs='+', default=[0, 1], help='List of valid detection classes.')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS threshold for filtering overlapping boxes.')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing image frames.')
    parser.add_argument('--model_type', type=str, choices=['MobileNetV2', 'EfficientNetB0'], default='EfficientNetB0', help='Choose the classifier model type.')
    parser.add_argument('--classifier_path', type=str, help='Path to the classifier model file.')

    args = parser.parse_args()

    classifier_config = ClassifierConfig(
        model_type=args.model_type,
        classifier_path=args.classifier_path if args.classifier_path else ('models/cnn/EfficientNetB0.h5' if args.model_type == 'EfficientNetB0' else 'models/cnn/EfficientNetB0.h5')
    )


    detection_config = DetectionConfig(
        model_path='models/yolo/best.pt',
        threshold_conf=args.threshold_conf,
        valid_classes=args.valid_classes,
        nms_threshold=args.nms_threshold
    )

    file_config = FileConfig(output_dir='assets/images_cropped')
    display_config = DisplayConfig(window_name="Vehicle Detection")

    detection_model = VehicleDetection(detection_config, classifier_config)
    if args.all_videos:
        if not os.path.isdir(args.video):
            print(f"Error: The directory '{args.video}' does not exist.")
            sys.exit(1)
        videos = path_videos(args.video)
        for video_path in videos:
            file_config = FileConfig(output_dir='assets/images_cropped')
            video_processor = VideoProcessor(
                video_path=video_path,
                detection_model=detection_model,
                file_config=file_config,
                display_config=display_config,
                frame_skip=args.frame_skip
            )
            video_processor.process_video()
    elif args.image_folder:
        if not os.path.isdir(args.image_folder):
            print(f"Error: The folder '{args.image_folder}' does not exist.")
            sys.exit(1)
        file_config = FileConfig(output_dir='assets/images_cropped')
        video_processor = VideoProcessor(
            video_path="",
            detection_model=detection_model,
            file_config=file_config,
            display_config=display_config,
            frame_skip=args.frame_skip
        )
        video_processor.process_image(args.image_folder)
    else:
        if not os.path.exists(args.video):
            print(f"Error: The video file '{args.video}' does not exist.")
            sys.exit(1)
        file_config = FileConfig(output_dir='assets/images_cropped')
        video_processor = VideoProcessor(
            video_path=args.video,
            detection_model=detection_model,
            file_config=file_config,
            display_config=display_config,
            frame_skip=args.frame_skip
        )
        video_processor.process_video()