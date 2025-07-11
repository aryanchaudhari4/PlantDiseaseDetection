# 🌿 Plant Disease Detection from Leaf Images

## 🎯 Objective
Detect and classify plant leaf diseases using image recognition and Convolutional Neural Networks (CNN).

## 🛠️ Tools & Technologies
- Python
- TensorFlow  Keras
- OpenCV
- Streamlit (for GUI)
- scikit-learn
- PlantVillage Dataset

## 🧠 Model Info
- 4 classes
  - Tomato___Early_blight
  - Tomato___Late_blight
  - Tomato___Leaf_Mold
  - Tomato___healthy
- Accuracy ~98% (train), ~91% (validation)
- Input Leaf image (JPGPNG)
- Output Predicted class with confidence

## 📸 Streamlit App
Upload an image and get instant prediction.

### ▶️ Run the App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
PlantDiseaseDetection
├── app.py
├── model
│   ├── plant_disease_model.h5
│   └── label_classes.pkl
├── dataset
│   └── (4 class folders)
├── notebooks
│   ├── data_preprocessing.ipynb
│   └── train_model.ipynb
├── requirements.txt
├── README.md
```

## 👨‍💻 Developed by Aryan Chaudhary
Intern at [Your Company Name]
