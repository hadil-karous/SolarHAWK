# ☀️ Solar Panel Anomaly Detection using YOLO11-OBB

This repository contains an end-to-end pipeline for detecting defects in solar panels using **Oriented Bounding Boxes (OBB)**. The project leverages **YOLO11m-OBB** and incorporates specialized image preprocessing to handle the low contrast and class imbalance often found in thermal and EL (Electroluminescence) solar imagery.

## 🚀 Key Features

* **YOLO11m-OBB Architecture:** Utilizes the latest Ultralytics model for detecting rotated objects, ensuring precise localization of panel defects.
* **CLAHE Enhancement:** Implements *Contrast Limited Adaptive Histogram Equalization* to sharpen detail in low-contrast solar images.
* **Automated Minority Augmentation:** A custom script that identifies rare defect classes (e.g., Reverse Polarity) and generates synthetic training examples using `Albumentations`.
* **Vegetation Box Refinement:** Includes a utility to tighten bounding boxes around vegetation to reduce background noise.
* **Google Colab Optimized:** Designed to run seamlessly with Google Drive integration and T4 GPU acceleration.

---

## 🛠️ Installation

```bash
pip install ultralytics albumentations opencv-python yaml
```

## 📂 Pipeline Workflow

### 1. Data Preparation
The notebook mounts Google Drive and updates the `data.yaml` file to ensure absolute paths are used for training in the Colab environment.

### 2. Preprocessing & Augmentation
* **OBB Tightening:** Shrinks vegetation bounding boxes by a factor of 0.90 to improve model focus.
* **Oversampling:** The script automatically creates 10 augmented copies of minority class images (using flips, rotations, and brightness shifts).
* **CLAHE Pipeline:** Converts the dataset into a high-contrast version saved as `dataset_clahe`.

### 3. Training
The model is trained with high-resolution settings (`imgsz=1024`) to capture tiny anomalies.

```python
results = model.train(
    data='path/to/dataset_clahe/data.yaml',
    epochs=50,
    imgsz=1024,
    optimizer='AdamW',
    lr0=0.001,
    project='YOLO_Results',
    name='solar_anomaly_final_run'
)
```

---

## 📊 Dataset Structure
The project expects a YOLO OBB formatted dataset:
```text
dataset/
├── train/
│   ├── images/
│   └── labels/ (class_id x1 y1 x2 y2 x3 y3 x4 y4)
├── valid/
└── data.yaml
```

## 📈 Results
Training logs, weights (`best.pt`), and validation plots are automatically saved to your Google Drive under the `/YOLO_Results/` directory.

---

## 📜 Credits
* **Model:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
* **Augmentation:** [Albumentations Library](https://albumentations.ai/)

---

