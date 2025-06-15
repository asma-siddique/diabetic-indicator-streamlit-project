
# 🩺 Diabetes Health Indicators Analysis

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

A machine learning pipeline to **predict diabetes risk** (No Diabetes / Prediabetes / Diabetes) using CDC's BRFSS 2015 dataset. The project includes full data exploration, model training, and deployment via a Streamlit web app.

---

## 🔍 Dataset Overview

- **Source:** [CDC BRFSS 2015 Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
- **Total Rows:** 253,680  
- **Target Variable:** `Diabetes_012`  
  - 0 = No Diabetes  
  - 1 = Prediabetes  
  - 2 = Diabetes  

### 🧾 Key Features

| Feature     | Description                    | Type       |
|-------------|--------------------------------|------------|
| HighBP      | High blood pressure            | Binary     |
| HighChol    | High cholesterol               | Binary     |
| BMI         | Body Mass Index                | Continuous |
| Smoker      | Smoking status                 | Binary     |
| GenHlth     | General health (1–5 scale)     | Ordinal    |
| PhysHlth    | Physical health days (0–30)    | Continuous |
| Age         | Age group (1–13 scale)         | Ordinal    |
| Income      | Income level (1–8 scale)       | Ordinal    |

---

## 🛠️ Project Pipeline

1. **Data Cleaning** – Missing values, duplicates, and outliers handled  
2. **Exploratory Data Analysis (EDA)** – 7+ interactive visualizations  
3. **Feature Engineering** – e.g., BMI categorization  
4. **Model Training** – Balanced Random Forest Classifier  
5. **Evaluation** – Accuracy, F1-score, class-wise metrics  
6. **Deployment** – Streamlit app  
7. **Insights** – Feature importance & risk interpretation  

---

## 📊 Exploratory Data Analysis

### 🔥 Correlation Heatmap
```python
plt.figure(figsize=(16,12))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

### 📈 Diabetes by Age Group (Plotly)
```python
px.bar(
    df.groupby('Age')['Diabetes_012'].value_counts(normalize=True).unstack(),
    title="Diabetes Prevalence by Age"
)
```

---

## 🤖 Model Training Example

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_scaled, y_train)
```

---

## 🚀 Deployment Guide

### 1. Install Dependencies
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn plotly joblib
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Visit the Dashboard  
🌐 [http://localhost:8501](http://localhost:8501)

---

## 🖼️ App Highlights

### 📂 Files Overview

| File/Folder         | Description                              |
|---------------------|------------------------------------------|
| `project.py`            | Main Streamlit application               |
| `diabetes_dataset.csv` | Raw BRFSS 2015 dataset              |
| `requirements.txt`  | List of Python dependencies              |
| `notebooks/`        | EDA and experimentation Jupyter files   |

---

### ✨ Features

#### ✅ EDA Dashboard
- 7+ interactive visualization types  
- Dynamic filtering & summaries  

#### 🧠 Prediction Interface
![Prediction Form](https://via.placeholder.com/800x400?text=Diabetes+Prediction+Form)
- 21 health inputs  
- Real-time prediction probabilities  
- Visualized risk  

#### 📉 Model Insights
- Feature importance plots  
- Threshold tuning  
- Class-wise metrics  

---

## 📌 Key Findings

### 🚨 Top Risk Factors:
- High blood pressure → **3.2× risk**
- Obesity (BMI > 30) → **2.8× risk**
- Poor general health → **4.5× risk**

### 🛡️ Protective Factors:
- Physical activity → **–38% risk**
- Normal BMI → **–52% risk**

### 🧑‍🤝‍🧑 Demographic Trends:
- Diabetes prevalence **increases with age**
- Higher rates in **low-income groups**

---

## 🔗 Resources

- 📓 [Kaggle Notebook: Diabetes Analysis](#) */kaggle/input/diabetes-risk-prediction*
 

---

## 👩‍💻 Author

**Name:** *Asma Siddique*  
**Institution:** *Punjab University*  
**GitHub:** [github.com/asma-siddique](https://github.com/asma-siddique/)  
**LinkedIn:** [asma-siddique-4389a22a6](www.linkedin.com/in/asma-siddique-4389a22a6)

---

## 📜 License

MIT License – see `LICENSE` file for details.

---



