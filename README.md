# 🩺 Breast Cancer Detection Using AI & Deep Learning

## 📌 Overview
This project utilizes **machine learning & deep learning** techniques to classify breast cancer images. It applies **image enhancement, segmentation, and classification** using the **CBIS-DDSM** and **Breast Histopathology Images** datasets.

## 🚀 Features
- **Image Enhancement**: Median Blur, Sharpening, CLAHE
- **Image Segmentation**: Watershed Algorithm & Canny Edge Detection
- **Image Classification**: CNN-based Model (99.67% Accuracy)
- **Data Visualization**: Histograms & Bar Graphs for CBIS-DDSM Data
- **Cancerous & Non-Cancerous Detection** using Histopathology Images

## 🛠️ Technologies Used
- **Python, TensorFlow, Keras, OpenCV**
- **Deep Learning Models: CNN**
- **Datasets: CBIS-DDSM & Breast Histopathology Images**
- **Visualization: Matplotlib, Pandas**
- **Jupyter Notebook for Development**

---

## 📂 Dataset Information
### **1. CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- **5GB Dataset** containing **Mammograms**
- Used for **single image classification** (enhancement, segmentation & classification)

### **2. Breast Histopathology Images**
- 162 whole-mount slide images scanned at **40x magnification**
- **277,524 image patches (50x50 px)**
- Used for **cancerous vs. non-cancerous classification**

🔹 **[Download Dataset](https://drive.google.com/your-link-here)**

---

## ⚡ Workflow
### **1️⃣ Image Enhancement**
✅ **Techniques Applied**:  
- **ROI Extraction** → Focuses on tumor regions  
- **Median Blur Filter** → Removes salt & pepper noise  
- **Sharpening Filter** → Enhances image clarity  
- **CLAHE (Contrast-Limited Adaptive Histogram Equalization)** → Increases contrast  

### **2️⃣ Image Segmentation**
✅ **Methods Used**:
- **Watershed Algorithm** → Separates foreground & background  
- **Canny Edge Detection** → Identifies edges of tumors  

### **3️⃣ Image Classification**
✅ **CNN Architecture (Trained Model)**
- **Watershed Segmented Image** → 99.67% Accuracy  
- **Canny Edge Segmented Image** → 97.82% Accuracy  

---

## 📊 Results & Visualizations
🔹 **Full Mammograms, Cropped & ROI Mask Images**  
🔹 **Malignant vs. Benign Tumors - Data Distribution**  
🔹 **Histopathology Image Classification (92.80% Accuracy)**  
🔹 **Bar Graphs & Pie Charts for Dataset Analysis**  

---

## 🔧 Installation & Setup
```bash
# Clone Repository
git clone https://github.com/yourusername/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection

# Install Dependencies
pip install -r requirements.txt

# Run the Model
python train.py
