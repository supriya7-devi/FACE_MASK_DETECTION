# 😷 Face Mask Detection using Deep Learning
## 📖 Overview
This project is a Deep Learning-based Face Mask Detection system that classifies faces into two categories: **With Mask** and **Without Mask**. It uses **transfer learning with MobileNetV2** for classification and **OpenCV's DNN face detector** to locate faces in real images before predicting, so it works on full photos — not just pre-cropped face images.
---
## 🎯 Objectives
- Detect faces in an image using a DNN-based face detector.
- Classify each detected face as **with mask** or **without mask**.
- Train a CNN model using transfer learning for high accuracy with limited data.
- Evaluate the model using standard classification metrics.
- Save the trained model for future use and inference on new images.
---
## 🗂️ Dataset
The dataset is provided as a zip file (uploaded via Google Colab) containing two classes:
```
dataset/
├── with_mask/
└── without_mask/
```
- Folder names are auto-detected with flexible matching (handles naming variations).
- Images are resized to **224 × 224 pixels** to match MobileNetV2's expected input.
- Data augmentation (rotation, shifts, shear, zoom, brightness, horizontal flip) is applied during training to improve generalization on a small dataset.
---
## 🛠️ Technologies Used
- Python
- Google Colab
- TensorFlow / Keras
- OpenCV (`cv2.dnn` for face detection)
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
---
## 📁 Project Structure
```
Face_Mask_Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── face_mask_detector.h5
│
├── outputs/
│
├── FACE_MASK_DETECTION_PROJECT.ipynb
│
├── requirements.txt
│
└── README.md
```
---
## ⚙️ Workflow
1. Upload and extract the dataset (`.zip`) in Google Colab.
2. Auto-detect the `with_mask` / `without_mask` folders and check class balance.
3. Preview sample images from each class.
4. Build augmented training/validation data pipelines using `ImageDataGenerator`.
5. Build the model using **MobileNetV2** as a frozen base, with a custom classification head (GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.4) → Dense(1, Sigmoid)).
6. Train the model with `EarlyStopping` and `ReduceLROnPlateau` callbacks.
7. Plot training/validation accuracy and loss curves.
8. Evaluate the model with a classification report and confusion matrix.
9. Save the trained model as `face_mask_detector.h5`.
10. Predict on new images: detect faces with OpenCV's DNN face detector (Caffe-based SSD model), then classify each detected face and draw labeled bounding boxes.
---
## 🤖 Machine Learning Approach
- **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet, base frozen)
- Binary classification head trained on top of the frozen base
- **OpenCV DNN face detector** (`res10_300x300_ssd_iter_140000.caffemodel`) used at inference time to locate faces before classification
---
## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training/validation accuracy & loss curves
---
## 🚀 Installation
Clone the repository:
```bash
git clone https://github.com/supriya7-devi/Face_Mask_Detection.git
```
Install the required libraries:
```bash
pip install -r requirements.txt
```
---
## ▶️ Run the Project
Open the notebook:
```
FACE_MASK_DETECTION_PROJECT.ipynb
```
Run all cells in Google Colab (recommended, for free GPU access) or Jupyter Notebook. When prompted, upload your dataset as a `.zip` file with `with_mask` / `without_mask` subfolders.
---
## 📈 Results
The trained model detects faces in an image and predicts whether each person is wearing a mask, drawing a labeled bounding box around each face.
Example:
```
Raw prediction value: 0.9860
MASK (98.6%)
```
or
```
Raw prediction value: 0.0220
NO MASK (97.8%)
```
---
## 🔮 Future Improvements
- Real-time webcam detection
- Fine-tuning MobileNetV2 layers instead of using a fully frozen base
- Mobile application integration
- CCTV surveillance support
- Web deployment using Flask or Streamlit
---
## 👩‍💻 Author
**Supriya**
Machine Learning Project
---
## 📄 License
This project is developed for educational and learning purposes.
