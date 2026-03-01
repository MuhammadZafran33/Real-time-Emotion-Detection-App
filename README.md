# 😄 Emotion Detection App

> **Real-time facial emotion recognition powered by DeepFace & Streamlit**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![DeepFace](https://img.shields.io/badge/DeepFace-Latest-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Working%20Locally-yellow?style=flat-square)

---

## 📌 Overview

The **Emotion Detection App** is a deep learning–powered web application that analyzes facial expressions from uploaded images and detects the dominant human emotion in real time. Built with **DeepFace** and **Streamlit**, it provides instant predictions with high accuracy across **7 core emotions**.

This is **Project 4** in my Fiverr ML Portfolio Series.

---

## 🎭 Detectable Emotions

| Emotion     | Emoji |
|-------------|-------|
| Happy       | 😊    |
| Sad         | 😢    |
| Angry       | 😠    |
| Surprised   | 😲    |
| Fear        | 😨    |
| Disgust     | 🤢    |
| Neutral     | 😐    |

---

## 🚀 Features

- 📸 **Upload any image** — supports JPG, JPEG, PNG
- 🧠 **DeepFace-powered analysis** — uses pre-trained deep learning models under the hood
- ⚡ **Instant results** — emotion predicted and displayed in seconds
- 🎨 **Clean, intuitive UI** — built with Streamlit for a smooth user experience
- 📊 **Confidence display** — shows the dominant emotion with visual feedback

---

## 🛠️ Tech Stack

| Tool         | Purpose                          |
|--------------|----------------------------------|
| Python       | Core programming language        |
| Streamlit    | Web app framework                |
| DeepFace     | Facial emotion recognition model |
| OpenCV       | Image preprocessing              |
| Pillow (PIL) | Image loading & handling         |
| NumPy        | Array operations                 |

---

## 📁 Project Structure

```
emotion-detection-app/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadZafran33/emotion-detection-app.git
cd emotion-detection-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
deepface
opencv-python
Pillow
numpy
tf-keras
```

---

## 💡 How It Works

1. **Upload** a clear photo of a face (JPG/PNG)
2. DeepFace **analyzes** the facial features using a pre-trained CNN model
3. The app **predicts** the dominant emotion from 7 categories
4. The result is **displayed** instantly with the emotion label and visual indicator

---

## 📸 Sample Output

```
Detected Emotion: HAPPY 😊
Confidence: High
```

---

## 🔗 Portfolio Links

| Project | App | GitHub |
|---------|-----|--------|
| Project 2: Diabetes Prediction | [Live App](https://zafii-diabetes-prediction.streamlit.app) | [GitHub](https://github.com/MuhammadZafran33/diabetes-prediction) |
| Project 3: Sentiment Analysis | [Live App](https://zafran-sentiment-analysis-app.streamlit.app) | [GitHub](https://github.com/MuhammadZafran33/sentiment-analysis-app) |
| Project 4: Emotion Detection | Local Only | [GitHub](https://github.com/MuhammadZafran33/emotion-detection-app) |

---

## 👨‍💻 Author

**Muhammad Zafran**
- 🌐 Fiverr: [muh_zafran](https://www.fiverr.com/muh_zafran)
- 💻 GitHub: [MuhammadZafran33](https://github.com/MuhammadZafran33)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ as part of the ML Streamlit Portfolio Series*
