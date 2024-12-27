
import cv2
import numpy as np
from .Configs import *
from tensorflow.keras.models import load_model
from ultralytics import YOLO

class VehicleDetection:
    def __init__(self, config: DetectionConfig, classifier_config: ClassifierConfig):
        self.model = YOLO(config.model_path)
        self.config = config
        self.conditions = DetectCondition()
        self.classifier_config = classifier_config
        self.classifier = load_model(self.classifier_config.classifier_path)
        self.image_size = (224, 224)

    def classify_damage(self, image):
        if self.classifier_config.model_type == 'MobileNetV2':
            self.image_size = (224, 224)
            image_resized = cv2.resize(image, self.image_size)  
            image_array = np.expand_dims(image_resized, axis=0)  
            image_array = image_array.astype('float32') / 255.0  
        elif self.classifier_config.model_type in ['EfficientNetB0', 'EfficientNetB3']:
            self.image_size = (224, 224)
            image_resized = cv2.resize(image, self.image_size)  
            image_array = np.expand_dims(image_resized, axis=0)  
            image_array = image_array.astype('float32')  

        # Predict the damage class
        prediction = self.classifier.predict(image_array)[0]  
        damage_class = np.argmax(prediction)  # Class index
        confidence = np.max(prediction)  
        
        return damage_class, confidence
    
    def CarConditionStatus(self, cls: int) -> str:
        return self.conditions.accident if cls == 0 else self.conditions.normal

    def process_frame(self, frame):
        results = self.model(frame)
        boxes, confidences, classes = [], [], []
        for result in results:
            if result.boxes:
                for detection in result.boxes:
                    conf = detection.conf.item()
                    cls = int(detection.cls.item())
                    if conf >= self.config.threshold_conf and cls in self.config.valid_classes:
                        x1, y1, x2, y2 = detection.xyxy[0].tolist()
                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        confidences.append(float(conf))
                        classes.append(str(cls))
        return boxes, confidences, classes


    def apply_nms(self, boxes, confidences):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.0, nms_threshold=self.config.nms_threshold)
        return [i[0] for i in indices] if len(indices) > 0 and isinstance(indices[0], list) else indices
