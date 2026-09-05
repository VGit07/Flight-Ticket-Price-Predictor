# ✈️ Flight Ticket Price Predictor

A Simple ML Web app that predicts the price of flight ticket based on source city, destination, the airlines you choose, duration of flight and class.

---

## 🎯 Features

* ✈️ It predicts flight ticket price
* 🏙️ Supports **6 Major Indian Cities**
* 🛫 Supports **Top Airlines in India**
* 🤖 **Random Forest** model is trained and tested with accuracy of ~92%
* 💻 **Streamlit** is used to make Web app and Deployment

---


## 📸 How it works ?

**You Give Inputs:**
* Source City
* Destination City
* Airlines
* Duration

**You Get :**
* Estimated Ticket Price from our trained model

---

## 📁 File Structure

```text
Flight-Ticket-Price-Predictor/
│
├── app.py                  # Streamlit App
├── requirements.txt         # Python dependencies
│
├── src/
│   └── train.py            # Model training script
│
├── data/
│   └── data.csv            # Original dataset
│   
│
├── model/
│   ├── model.pkl           # Trained ML model
│   └── encoder.pkl         # Feature encoder
│
└── notebook/
    ├── EDA.ipynb           # Exploratory Data Analysis
    └── model.ipynb         # For Model Analysis and Development
```

---

## 🚀 How to install

### 1. Clone the Repository

```bash
git clone https://github.com/VGit07/Flight-Price-Predictor
cd Flight-Price-Predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```


---
## 🤖 About the Model
---
* Machine Learning model used : Random Forest
* Training : Testing split = 8:2
* Accuracy of the model : ~92%
---
## 📊 Model Performance

| Metric           |        Result |
| ---------------- | ------------: |
| **R² Score**     |         ~0.92 |
| **MAE**          | ~₹1,500–₹2,000|
| **Dataset Size** |  10K+ records |

---

## 📦 Dependencies

The project uses:

```text
pandas
numpy
scikit-learn
streamlit
prettytable
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Tech Stack

* 🐍 **Python**
* 🤖 **Scikit-learn**
* 🎨 **Streamlit**
* 📊 **Pandas**
* 🔢 **NumPy**
* 📓 **Jupyter Notebook**

### EDA & Analysis

EDA and Analysis were done by :
* Claude 🤖
* ChatGPT ✨

---

## 📊 Dataset

The project uses an already existing clean and real data from **Kaggle**

---

## 👨‍💻 Author

### Made by: Vignesh Nadar

---

## 🌐 Live Demo
**Link:** https://ft-price-predictor.streamlit.app/

---
