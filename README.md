# AutoJudge – Programming Task Difficulty Estimation

AutoJudge is a machine learning based system designed to estimate the difficulty of competitive programming tasks using their problem statements. Given a task description as input, the system predicts:
- 🏷️ A **categorical difficulty level**: Easy / Medium / Hard  
- 📊 A **numerical difficulty score**

The project combines **natural language processing techniques** with **classical machine learning models** and provides a simple **Streamlit-based web interface** for interactive evaluation.

---

## 📌 Project Overview

Difficulty classification of programming problems is a key requirement for competitive programming platforms and learning systems. Manual difficulty tagging is often subjective and inconsistent.

AutoJudge aims to automate this process by:
- Learning patterns directly from problem statements  
- Extracting meaningful text-based features  
- Using trained machine learning models for prediction  

The system runs completely locally and includes:
- A full preprocessing pipeline  
- Trained classification and regression models  
- A web-based interface for testing predictions  

---

## 📂 Dataset Used

The dataset used in this project is **TaskComplexityEval-24**, which was officially provided by the club.

- The dataset consists of competitive programming problem statements labeled with difficulty-related information.
- It is intended for evaluating task complexity estimation models.

📁 **Raw dataset files:**  
- `data/raw/`

📁 **Processed dataset files:**  
- `data/processed/`

During preprocessing, raw data is cleaned and transformed into structured formats suitable for feature extraction and model training.

---

## 🧠 Approach and Models Used

### 🔹 Preprocessing
- Cleaning and normalization of problem statements  
- Removal of unnecessary symbols and formatting  
- Conversion into structured datasets  

### 🔹 Feature Extraction
- TF-IDF vectorization of problem statements  
- Additional handcrafted features:
  - Statement length  
  - Keyword frequency  
  - Symbol density  

### 🔹 Models
- **Classification:** RandomForestClassifier  
  - Predicts difficulty category (Easy / Medium / Hard)
- **Regression:** GradientBoostingRegressor  
  - Predicts a continuous difficulty score  

---

## 📊 Evaluation Metrics

Model performance was evaluated using standard metrics:
- 🎯 **Classification:** Accuracy  
- 📉 **Regression:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)  

These metrics were computed on a validation split during training.

---

## 💾 Trained Models

All trained models are saved and included in the repository.

📁 **Location:**  
- `data/processed/`

📄 **Files included:**
- `classifier.pkl` – difficulty classification model  
- `regressor.pkl` – difficulty regression model  
- `tfidf.pkl` – TF-IDF vectorizer  

---

## 🖥️ Web Interface

The project includes a **Streamlit-based web interface** that allows users to:
- Enter a programming problem statement  
- View the predicted difficulty category  
- View the corresponding difficulty score  

The interface runs locally and does not require deployment or hosting.

---

## ▶️ Steps to Run the Project Locally

1. Clone the repository and navigate to the project directory  
2. Create a virtual environment using:
   - `python -m venv venv`
3. Activate the virtual environment (platform-specific)
4. Install required libraries:
   - `pip install streamlit pandas numpy scikit-learn joblib`
5. (Optional) Run the full machine learning pipeline:
   - `python src/preprocess.py`
   - `python src/features.py`
   - `python src/train.py`
6. Start the web application:
   - `streamlit run app.py`
7. Open the application in the browser at:(The application will open automatically in the default browser at http://localhost:8501. If it does not open automatically, the URL can be accessed manually.)
   - `http://localhost:8501`

---

## 🎥 Demo Video

A short demo video demonstrating the project overview, model approach, and working web interface is provided.

🔗 **Demo video link:**  
- ADD DEMO VIDEO LINK HERE

---
## 📄 Project Report

A detailed project report (PDF) describing the problem statement, dataset, preprocessing steps, feature engineering techniques, model design, experimental setup, evaluation metrics (accuracy, confusion matrix, MAE, RMSE), web interface, and sample predictions is provided below.

Project Report (PDF):
https://drive.google.com/file/d/1GZpRHHj01orli5JA1Jb6ozuVJi2MtZwH/view?usp=sharing

The report corresponds directly to the submitted codebase and includes screenshots of outputs and the web interface.

---
## 👤 Author

**Darabala Sohan Mahendra**  
B.Tech, Computer Science  
Indian Institute of Technology Roorkee
