![Logo](https://uni-obuda.hu/wp-content/uploads/2021/11/kep3.jpg)
# 🎵 Music Genre Classification with Machine Learning

**Status:** ✅ Completed  
**University Project** – Óbuda University, BSc in Computer Engineering  
**Subject:** Machine Learning / AI specialization

## 📘 Overview

This project focuses on the automatic classification of music tracks into predefined **genres** using machine learning techniques.

It combines **audio signal processing** with **metadata from the Spotify API** to improve genre classification performance.

The main goal is to demonstrate how artificial intelligence can be applied to multimedia data, and how open APIs like Spotify’s can enhance traditional audio-based models.

## 🧠 Technologies Used

### Core

- **Python** (main language)
- **Librosa** – audio feature extraction
- **NumPy**, **Pandas**, **Scikit-learn** – data processing and ML
- **TensorFlow** / **Keras** – deep learning model

### External Data

- **Spotipy** – a lightweight Python client for the **Spotify Web API**, used to fetch:
  - Track metadata
  - Genre tags
  - Artist information

### Optional

- **Streamlit** – for interactive web-based demo (not yet implemented)

## 🎓 Educational Context

This project was developed as part of the **AI specialization** in the **Computer Engineering BSc program** at **Óbuda University**, for a machine learning course.

It shows how combining signal processing with real-world metadata can improve music classification systems.

## 📁 Repository Contents

- `genre_classification.ipynb` – Main Jupyter Notebook  
- `models/` – Trained models (optional)  
- `data/` – Audio and metadata inputs (not included)  
- `README.md` – Project overview  

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/MisurdaLuca/Genre_classification.git
   cd Genre_classification

> **Note**: Due to copyright and file size constraints, the dataset may not be included. Please add your own dataset (e.g., GTZAN) if needed.
