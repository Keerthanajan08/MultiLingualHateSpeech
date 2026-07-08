# 🛡️ Multimodal Hate Speech Detection using Fine-Tuned LLaMA 2

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)
![License](https://img.shields.io/badge/License-MIT-green)
![IEEE](https://img.shields.io/badge/Publication-IEEE-blue)

An **IEEE-published** multimodal AI system for detecting hate speech across **text, audio, and video** using **fine-tuned LLaMA 2**, **OpenAI Whisper**, and **Google Speech Recognition**.

The project combines Natural Language Processing (NLP), Speech Recognition, and Large Language Models (LLMs) into a unified content moderation pipeline capable of processing multilingual user-generated content.

---

# 🚀 Overview

Traditional hate speech detection systems focus only on textual content.

This project extends content moderation by supporting multiple input modalities, allowing users to submit:

- 📝 Text
- 🎙️ Audio
- 🎥 Video

Audio and video inputs are automatically transcribed before being classified using a fine-tuned LLaMA 2 model.

---

# 🏗️ System Architecture

<p align="center">
<img src="/UML.PNG" width="900">
</p>

---

# ✨ Features

- 📝 Multilingual text classification
- 🎙️ Audio transcription using Google Speech Recognition
- 🎥 Video transcription using OpenAI Whisper
- 🤖 Fine-tuned LLaMA 2 for hate speech classification
- 🌍 Multilingual support
- 📊 End-to-end multimodal inference pipeline
- 📈 IEEE-published research implementation

---

# 📂 Repository Structure

```text
MultiLingualHateSpeech/
│
├── app.py
├── requirements.txt
├── training1.ipynb
├── README.md
│
├── models/
│
├── datasets/
│
├── hatespeech/
│
├── images/
│   ├── architecture.png
│   ├── ui.png
│   └── results.png
│
└── outputs/
```

---

# 🛠️ Tech Stack

## Programming

- Python

## Machine Learning

- LLaMA 2
- Hugging Face Transformers
- PyTorch
- Scikit-learn

## Speech Processing

- OpenAI Whisper
- Google Speech Recognition API

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Keerthanajan08/MultiLingualHateSpeech.git
cd MultiLingualHateSpeech
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

---

# 💻 Usage

The application accepts three different input types.

### 📝 Text

Enter any text and receive a hate speech prediction.

---

### 🎙️ Audio

Upload an audio file.

The system:

1. Converts speech into text
2. Sends transcription to the classifier
3. Predicts Hate / Non-Hate

---

### 🎥 Video

Upload a video.

The pipeline:

Video

↓

OpenAI Whisper

↓

Speech Transcript

↓

Fine-Tuned LLaMA 2

↓

Prediction

---

# 📊 Results

| Metric | Value |
|---------|------:|
| Accuracy | **97%** |
| Model | Fine-Tuned LLaMA 2 |
| Input Modalities | Text, Audio, Video |
| Languages | Multilingual |

---

# 📸 Screenshots

## User Interface

<p align="center">
<img src="images/ui.png" width="900">
</p>

---

## Sample Prediction

<p align="center">
<img src="images/results.png" width="900">
</p>

---

# 📖 Publication

**Multimodal Hate Speech Detection using Fine-Tuned LLaMA 2**

Published in **IEEE Xplore (IACIS 2024).**

> This repository contains the implementation of the research presented in the publication.

---

# 🔬 Future Work

- Support for Instagram Reels
- Support for YouTube links
- Support for X (Twitter) videos
- Real-time moderation
- Explainable AI predictions
- REST API
- Docker deployment
- Cloud deployment
- Streaming inference
- Larger multilingual datasets

---

# 🤝 Contributing

Contributions are welcome.

If you find a bug or have suggestions for improvement, feel free to open an issue or submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👩‍💻 Author

**Keerthana Sasidaran**

📧 Email: your-email

🔗 LinkedIn: your-linkedin

💻 GitHub: https://github.com/Keerthanajan08
