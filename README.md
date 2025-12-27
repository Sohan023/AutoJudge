# AutoJudge: Hybrid NLP Difficulty Estimator 🧠⚡

**AutoJudge** is an intelligent difficulty prediction system for competitive programming problems. Unlike standard classifiers, it employs a **Hybrid Architecture** that combines Machine Learning (Random Forest + TF-IDF) with **Rule-Based Heuristics** to achieve high accuracy in categorizing problems as *Easy*, *Medium*, or *Hard*.



[Image of machine learning pipeline diagram]


## 🚀 Key Features
* **Dual-Model Engine:**
  * **Classifier:** `RandomForestClassifier` (n=800) for categorical difficulty (Easy/Med/Hard).
  * **Regressor:** `GradientBoostingRegressor` for precise difficulty scoring (1-10 scale).
* **NLP Feature Engineering:** Extracts **TF-IDF n-grams** combined with custom features like "Math Symbol Density" and "Algorithmic Keyword Frequency" (e.g., detecting 'dp', 'segment tree').
* **Hybrid Stratification:** A post-processing layer that uses **Expert Rules** to correct model biases (e.g., forcing 'Hard' classification if 'FFT' or 'Bitmask' is detected).
* **Modern UI:** A fully responsive **Streamlit** web application with a custom Glassmorphism dark theme.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python), Custom CSS
* **NLP & ML:** Scikit-Learn (TF-IDF, Random Forest, Gradient Boosting), NumPy, Pandas
* **Persistence:** Joblib for model serialization

## 📊 System Architecture
```text
Input (Problem Text) 
   │
   ├──> [TF-IDF Vectorizer] ──> Sparse Matrix
   │
   ├──> [Feature Extractor] ──> (Len, Math_Symbols, Keyword_Count)
   │
   ▼
[Merged Feature Vector]
   │
   ├──> [Random Forest Classifier] ──> Raw Class (0,1,2)
   │
   ├──> [Gradient Boosting Regressor] ──> Raw Score
   │
   ▼
[Rule-Based Stratification Layer] <── (Overrides edge cases)
   │
   ▼
Final Output: "Hard" (Score: 8.5/10)
