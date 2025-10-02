# Multilingual Hate Speech Detection

A Multilingual Hate Speech Detection System that supports text, audio, and video inputs. It uses fine-tuned transformer models (Llama 2 / HuggingFace) along with speech-to-text pipelines to classify content as hate speech or non-hate speech.

## ✨ Features

🔤 Text Classification – Detects hate speech in multilingual text

🎙️ Audio Support – Converts speech to text via Google Speech Recognition API

🎥 Video Support – Uses Whisper to transcribe video / YouTube content before classification

📊 Performance – Achieved 97% accuracy on benchmark dataset

🌍 Multilingual – Supports multiple languages

## 🛠️ Tech Stack

Python 3.9+

PyTorch, HuggingFace Transformers

Google Speech Recognition API

OpenAI Whisper

Pandas, NumPy, Scikit-learn

Matplotlib / Seaborn

## Getting Started

1️⃣ Clone the Repository
git clone https://github.com/username/MultiLingualHateSpeech.git
cd MultiLingualHateSpeech

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
python app.py

4️⃣ Run Training

Open and execute training1.ipynb in Jupyter Notebook or VS Code.
   
## 📊 Results

Accuracy: 97%

Metrics: Precision, Recall, F1 (see notebooks)

Model Files: Stored in hatespeech/

## 🔮 Future Improvements

Fine-grained categories (offensive, abusive, neutral)

Deploy as a REST API / Web App

Real-time streaming moderation

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📜 License

Licensed under the MIT License.
