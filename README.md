# 🛩️ Aerial Object Classification — Streamlit Web App  
### **Bird vs Drone Classifier using Deep Learning**

This project provides a **Streamlit-based web application** that classifies aerial objects (Bird / Drone) using a fine-tuned deep learning model (EfficientNet-B0 / SimpleCNN).  
The model was trained on a high-quality dataset of aerial images and deployed using Streamlit Cloud.

---

## 🚀 Features
- ✔️ **Upload any aerial image** (JPG/PNG)  
- ✔️ **Predict Bird or Drone**  
- ✔️ **Displays prediction probability**  
- ✔️ **Visual bar chart of class probabilities**  
- ✔️ **Grad-CAM heatmap** (model explainability)  
- ✔️ Lightweight and deployable via **Streamlit Cloud**  
- ✔️ Supports CPU—inference runs fast  

---

## 📁 Repository Structure

```
aerial-classification-streamlit/
├── app.py                          # Streamlit application
├── best_model_for_streamlit.pth    # Trained model checkpoint
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🧠 Model Details
The application automatically loads the trained model from  
`best_model_for_streamlit.pth`.

Two model types are supported:

### **1. EfficientNet-B0 (Transfer Learning)**
- Pretrained on ImageNet  
- Final classifier retrained on aerial dataset  
- High accuracy & excellent generalization  

### **2. SimpleCNN (Custom Model)**
- Lightweight convolutional neural network  
- Suitable for demonstrations or low-resource environments  

The app detects which architecture your checkpoint belongs to and loads it automatically.

---

## 📦 Installation (Local)

Clone the repository:

```bash
git clone https://github.com/<your-username>/aerial-classification-streamlit.git
cd aerial-classification-streamlit
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run Streamlit locally:

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

---

## 🌐 Deploying to Streamlit Cloud

1. Go to **https://share.streamlit.io**  
2. Sign in with GitHub  
3. Click **New App**  
4. Select:
   - Repository: `aerial-classification-streamlit`
   - Branch: `main`
   - File: `app.py`
5. Click **Deploy**

Your app will launch within seconds.

---

## 🎯 How It Works
1. User uploads an image  
2. Image is resized & normalized  
3. Model performs inference  
4. Softmax probabilities returned  
5. Grad-CAM heatmap is generated (if supported by architecture)  
6. Prediction + probability + heatmap shown in UI  

---

## 📸 Example Output
**Prediction: Bird**  
**Probability: 0.982**  
Bar chart shows class confidence, and Grad-CAM highlights regional focus (wings / body).

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- Torchvision  
- Streamlit  
- Grad-CAM  
- NumPy / OpenCV  
- PIL (Pillow)

---

## 📊 Dataset Summary
- Two classes: **Bird**, **Drone**  
- Train/Valid/Test split  
- ~3300 total images  
- Image size normalized to 224×224  
- Augmentations: rotation, flipping, color jitter  

---

## ✨ Author
**Charan**  
AI & Deep Learning Engineer

If you found this project useful, ⭐ the repo!

---

## 📬 Contact
Feel free to open an issue or message for suggestions, improvements, or collaborations.
