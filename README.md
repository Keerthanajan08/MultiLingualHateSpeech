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

## Similarity Score : 

   How does it decide which item is most similar to the item user likes? Here we use the similarity scores.
   
   It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.
   
## How Cosine Similarity works?
  Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.
  
  ![image](https://user-images.githubusercontent.com/36665975/70401457-a7530680-1a55-11ea-9158-97d4e8515ca4.png)

  
More about Cosine Similarity : [Understanding the Math behind Cosine Similarity](https://www.machinelearningplus.com/nlp/cosine-similarity/)

### Sources of the datasets 

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
3. [List of movies in 2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018)
4. [List of movies in 2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)
5. [List of movies in 2020](https://en.wikipedia.org/wiki/List_of_American_films_of_2020)

