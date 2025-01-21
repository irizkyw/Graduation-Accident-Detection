from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DetectionConfig:
    model_path: str
    threshold_conf: float
    valid_classes: List[int]
    nms_threshold: float = 0.4

@dataclass
class FileConfig:
    output_dir: str
    image_size: Tuple[int, int] = (224, 224)
    frame_prefix: str = 'frame'
    track_prefix: str = 'track'
    video_folder: str = ''

    def OutFileName(self, frame_count: int, track_id: int) -> str:
        return f"{self.output_dir}/{self.video_folder}/{self.frame_prefix}_{frame_count}_{self.track_prefix}_{track_id}.jpg"


@dataclass
class DisplayConfig:
    window_name: str = "Window"
    overlay_alpha: float = 0.3
    overlay_text: str = "Accident Detected!"
    WINDOW_X: int = 1280
    WINDOW_Y: int = 720

@dataclass
class DetectCondition:
    accident: str = "accident"
    normal: str = "no accident"

@dataclass
class ClassifierConfig:
    model_type: str = 'EfficientNetB0'  
    classifier_path: str = 'models/cnn/EfficientNetB0.h5'