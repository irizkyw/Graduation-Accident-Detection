
# Real-Time Vehicle Damage Classification Based on Accident Detection From CCTV Footage Using Two-Stage Approach

### Project Overview

This project presents a **two-stage approach** for **real-time vehicle damage classification** from CCTV footage. It integrates **accident detection** and **damage classification** using **YOLOv8** for accident detection and a **Convolutional Neural Network (CNN)** for damage classification.

<p align="center">
  <img src="https://github.com/irizkyw/Graduation-Accident-Detection/blob/main/assets/demo-gif.gif" alt="Demo">
</p>

### Key Features

- **Real-time accident detection** from CCTV footage.
- **Vehicle damage classification** based on severity (normal, moderate, severe).
- **Two-stage approach**:
  1. **Accident Detection** using YOLOv8.
  2. **Damage Classification** using Convolutional Neural Network (CNN).

### Technologies Used

- **YOLOv8**: For real-time object detection, particularly focused on identifying vehicles involved in accidents.
- **Convolutional Neural Networks (CNN)**: For classifying vehicle damage.
- **Python**: The primary programming language.
- **OpenCV**: For video processing and frame extraction.
- **TensorFlow/PyTorch**: For model development and training.

### Library Versions

- **ultralytics**: 8.3.36
- **torchvision**: 0.20.1+cu124
- **tensorflow**: 2.15.0
- **opencv-python**: 4.10.0.84

### Dataset

You can contribute to improving the scalability and accuracy of the model by sharing relevant datasets. Any dataset related to vehicle accidents, damage severity, or vehicle images can help enhance the performance and robustness of the system. To contribute, you can visit [released page](https://github.com/irizkyw/Graduation-Accident-Detection/releases/tag/1.0), or feel free to contact us for further collaboration and dataset sharing.
### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/irizkyw/Graduation-Accident-Detection.git
    cd Graduation-Accident-Detection
    ```

2. Create a Conda environment for installation. For detailed installation instructions, refer to the [Ultralytics Documentation](https://docs.ultralytics.com/quickstart/#install-ultralytics):

    ```bash
    conda create --name vehicle_damage_env python=3.9.19
    conda activate vehicle_damage_env
    ```

### Usage

Run the integrated system combining both stages (accident detection and damage classification) with the following command:

```bash
python main.py --model_type="MobileNetV2" --classifier_path="model\MobileNetV2.h5" --video="assets\videos\Testing Video.ts"
```

### Arguments for `main.py`:

- `--video`: Path to the video file or directory.
- `--all_videos`: Process all videos in the specified folder and subfolders.
- `--threshold_conf`: Confidence threshold for detections (default: 0.7).
- `--frame_skip`: Number of frames to skip between processing (default: 1).
- `--valid_classes`: List of valid detection classes (default: `[0, 1]`).
- `--nms_threshold`: NMS threshold for filtering overlapping boxes (default: 0.4).
- `--image_folder`: Path to the folder containing image frames.
- `--model_type`: Choose the classifier model type (`MobileNetV2` or `EfficientNetB0`, default: `EfficientNetB0`).
- `--classifier_path`: Path to the classifier model file.

### Pre-trained Models

Download the pre-trained YOLOv8 model for accident detection and the CNN model for damage classification from the `models/` directory.

- YOLOv8 weights for accident detection: `models/yolov8_model/`
- CNN model for damage classification: `models/cnn_model/`

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Citation
If you find our dataset/model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！
```bib
coming soon!
```
