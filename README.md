🗣️ Multilingual Hate Speech Detection

This project implements a Multilingual Hate Speech Detection System that supports text, audio, and video inputs. It leverages fine-tuned transformer models (Llama 2 / HuggingFace) along with speech-to-text pipelines to classify content as hate speech or non-hate speech.

📂 Project Structure
MultiLingualHateSpeech/
│
├── hatespeech/                # Fine-tuned model files
│   ├── config.json
│   ├── pytorch_model.bin       # Trained model weights (LFS recommended)
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── app.py                      # Main application script
├── training1.ipynb             # Training notebook
├── test.ipynb                  # Testing / evaluation notebook
│
├── balanced_hatespeech_dataset.xlsx   # Dataset 1
├── balanced_hatespeech_dataset1.xlsx  # Dataset 2
├── labeled_data.xlsx                  # Additional dataset
│
├── data/                      # JSON dataset / preprocessing files
├── demo.mp4                   # Demo video
├── temp.mp4                   # Temporary testing video
├── WhatsApp Video.mp4          # Sample input
│
└── README.md                   # Documentation

✨ Features

🔤 Text Classification – Detects hate speech in multilingual text.

🎙️ Audio Support – Converts speech to text via Google Speech Recognition API.

🎥 Video Support – Uses Whisper to transcribe video / YouTube content before classification.

📊 Performance – Achieved 97% accuracy on benchmark dataset.

🌍 Multilingual – Supports multiple languages (via pretrained tokenizer + multilingual datasets).

🛠️ Tech Stack

Python 3.9+

PyTorch, Transformers (HuggingFace)

Google Speech Recognition API

OpenAI Whisper

Pandas, NumPy, Scikit-learn

Matplotlib / Seaborn

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/username/MultiLingualHateSpeech.git
cd MultiLingualHateSpeech

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
python app.py

4️⃣ Run Training

Open and execute training1.ipynb in Jupyter Notebook or VS Code.

📊 Results

Accuracy: 97%

Metrics: Precision, Recall, F1 available in notebook outputs

Model files stored in hatespeech/

🔮 Future Improvements

Extend to more fine-grained categories (abusive, offensive, neutral).

Deploy as a REST API / Web App.

Integrate with real-time moderation systems.

🤝 Contributing

Contributions, issues, and feature requests are welcome!

📜 License

This project is licensed under the MIT License.
