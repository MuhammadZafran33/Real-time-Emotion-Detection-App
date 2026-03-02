<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Emotion%20Detection%20App&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Real-Time%20Facial%20Emotion%20Recognition%20%7C%20DeepFace%20%26%20Streamlit&descAlignY=58&descSize=18"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeepFace-Latest-00C851?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Status-Working%20Locally-FFD700?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Portfolio-Project%204-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Emotions-7%20Classes-FF6B6B?style=for-the-badge"/>
</p>

<br/>

> 🧠 *A deep learning–powered web app that reads your face and understands how you feel — in seconds.*

</div>

---

## 🌟 What Is This?

<table>
<tr>
<td width="60%">

The **Emotion Detection App** is a production-ready deep learning web application that analyzes facial expressions from uploaded images and detects the dominant human emotion in **real time**.

Built on **DeepFace** — one of the world's most accurate facial analysis frameworks — and wrapped in a clean **Streamlit** interface, this app brings state-of-the-art computer vision to your browser instantly.

This is **Project 4** of my **Fiverr ML Portfolio Series** — designed to showcase real-world AI capabilities to clients worldwide.

</td>
<td width="40%" align="center">
```
┌─────────────────────────┐
│   📸  Upload Photo      │
│         ↓               │
│   🧠  DeepFace CNN      │
│         ↓               │
│   🎯  Emotion Detected  │
│                         │
│   😊  HAPPY — 94.3%     │
└─────────────────────────┘
```

</td>
</tr>
</table>

---

## 🎭 Detectable Emotions

<div align="center">

| Emotion | Emoji | Description |
|:-------:|:-----:|:-----------:|
| **Happy** | 😊 | Joy, excitement, positivity |
| **Sad** | 😢 | Grief, sorrow, unhappiness |
| **Angry** | 😠 | Frustration, rage, displeasure |
| **Surprised** | 😲 | Shock, amazement, wonder |
| **Fear** | 😨 | Anxiety, fright, apprehension |
| **Disgust** | 🤢 | Revulsion, aversion, distaste |
| **Neutral** | 😐 | Calm, expressionless, composed |

</div>

---

## ✨ Features

<div align="center">

| Feature | Description |
|:-------:|-------------|
| 📸 **Multi-format Upload** | Supports JPG, JPEG, and PNG image formats |
| 🧠 **DeepFace Engine** | Powered by pre-trained deep CNN models (VGG-Face, FaceNet) |
| ⚡ **Instant Prediction** | Emotion analyzed and displayed in under 2 seconds |
| 🎨 **Modern UI** | Clean, responsive Streamlit interface with visual feedback |
| 📊 **Confidence Score** | Shows dominant emotion with probability percentage |
| 🔒 **Privacy First** | No images stored — all processing done locally |

</div>

---

## 🛠️ Tech Stack
```
╔══════════════════════════════════════════════════════════════╗
║                    TECHNOLOGY STACK                          ║
╠══════════════╦═══════════════════════════════════════════════╣
║  Python      ║  Core language — logic & data processing      ║
║  Streamlit   ║  Web app framework — UI & deployment          ║
║  DeepFace    ║  Facial emotion recognition (CNN models)       ║
║  OpenCV      ║  Image loading, preprocessing, manipulation    ║
║  Pillow      ║  Image format handling & conversion            ║
║  NumPy       ║  Numerical operations & array management       ║
║  TF-Keras    ║  Deep learning backend for DeepFace            ║
╚══════════════╩═══════════════════════════════════════════════╝
```

---

## 📁 Project Structure
```
📦 emotion-detection-app/
│
├── 📄 app.py                  ← Main Streamlit application
├── 📋 requirements.txt        ← Python dependencies
└── 📖 README.md               ← You are here!
```

---

## ⚙️ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MuhammadZafran33/emotion-detection-app.git
cd emotion-detection-app
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the App 🚀
```bash
streamlit run app.py
```

> 🌐 Opens at **`http://localhost:8501`**

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
```
 STEP 1        STEP 2            STEP 3           STEP 4
────────     ──────────       ──────────        ──────────
📸 Upload → 🧠 DeepFace  →  🎯 Dominant   →  📊 Display
  Photo       CNN Analysis     Emotion           Result
 (JPG/PNG)   (VGG / FaceNet)   Identified        + Score
```

1. **Upload** a clear front-facing photo (JPG or PNG)
2. **DeepFace** processes facial geometry using a pre-trained CNN
3. **Model** classifies expression into one of 7 emotion categories
4. **Result** displayed instantly with confidence score

---

## 📸 Sample Output
```
┌────────────────────────────────────────┐
│        🎭 EMOTION ANALYSIS RESULT      │
├────────────────────────────────────────┤
│                                        │
│   Detected Emotion :  HAPPY  😊        │
│   Confidence       :  94.3%            │
│   Model Used       :  VGG-Face         │
│   Processing Time  :  1.2 seconds      │
│                                        │
└────────────────────────────────────────┘
```

---

## 🔗 ML Portfolio — All Projects

<div align="center">

| # | Project | Live App | GitHub | Status |
|:-:|---------|:--------:|:------:|:------:|
| 01 | 🩺 Diabetes Prediction | [![Live](https://img.shields.io/badge/▶_Live-App-FF4B4B?style=flat-square&logo=streamlit)](https://zafii-diabetes-prediction.streamlit.app) | [![GitHub](https://img.shields.io/badge/Code-black?style=flat-square&logo=github)](https://github.com/MuhammadZafran33/diabetes-prediction) | ✅ Deployed |
| 02 | 💬 Sentiment Analysis | [![Live](https://img.shields.io/badge/▶_Live-App-FF4B4B?style=flat-square&logo=streamlit)](https://zafran-sentiment-analysis-app.streamlit.app) | [![GitHub](https://img.shields.io/badge/Code-black?style=flat-square&logo=github)](https://github.com/MuhammadZafran33/sentiment-analysis-app) | ✅ Deployed |
| 03 | 😄 Emotion Detection | 🟡 Local Only | [![GitHub](https://img.shields.io/badge/Code-black?style=flat-square&logo=github)](https://github.com/MuhammadZafran33/emotion-detection-app) | 🟡 Local |

</div>

---

## 👨‍💻 About the Author

<div align="center">
```
╔════════════════════════════════════════════════╗
║           Muhammad Zafran                      ║
║     AI & Machine Learning Engineer             ║
║     BSAI Student — IM|Sciences, Peshawar       ║
╠════════════════════════════════════════════════╣
║  🌐 Fiverr  →  fiverr.com/muh_zafran           ║
║  💻 GitHub  →  github.com/MuhammadZafran33      ║
║  📍 Location → Peshawar, KP, Pakistan           ║
╚════════════════════════════════════════════════╝
```

<p>
  <a href="https://www.fiverr.com/muh_zafran">
    <img src="https://img.shields.io/badge/Hire%20Me%20on-Fiverr-1DBF73?style=for-the-badge&logo=fiverr&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/MuhammadZafran33">
    <img src="https://img.shields.io/badge/Follow%20on-GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</p>

</div>

---

## 📄 License
```
MIT License — Free to use, modify, and distribute with attribution.
Copyright (c) 2025 Muhammad Zafran
```

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling"/>

**⭐ Star this repo if you found it useful — it motivates me to build more! ⭐**

*Built with ❤️ from Peshawar, Pakistan — ML Streamlit Portfolio Series*

</div>
