# 🎓 Student Performance Predictor Using ANN

### Predicting Academic Performance with Artificial Intelligence

An end-to-end Deep Learning project that uses an **Artificial Neural Network (ANN)** to predict students' average examination scores based on demographic, parental education, lunch, and test preparation information.

The project combines a trained ANN model with an interactive **Streamlit web application**, allowing users to enter student information and receive an estimated academic performance score.

---

## 🚀 Project Overview

Student academic performance can be influenced by several factors, including parental education, access to resources, and test preparation.

This project explores how Artificial Neural Networks can learn patterns from student-related data and estimate average examination performance.

Users can enter student details through a simple web interface and receive a predicted average score out of 100.

### ✨ Key Features

- Artificial Neural Network (ANN) for regression.
- Data preprocessing using Scikit-learn.
- One-Hot Encoding for categorical features.
- Real-time predictions through Streamlit.
- Interactive and responsive user interface.
- Saved model and preprocessing pipeline for reuse.
- Prediction score visualization.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | ANN model development |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing and encoding |
| Joblib | Saving the preprocessing pipeline |
| Dill | Serialization dependency |
| Streamlit | Web application |
| Matplotlib | Data visualization (if used in notebook) |

---

## 📊 Dataset

**Dataset:** Students Performance in Exams

**Source:** [Kaggle – Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

The dataset contains 1,000 student records with the following attributes:

| Feature | Description |
|---|---|
| Gender | Student gender |
| Race/Ethnicity | Student group |
| Parental Level of Education | Parents' education level |
| Lunch | Standard or free/reduced lunch |
| Test Preparation Course | Whether the student completed test preparation |
| Math Score | Mathematics examination score |
| Reading Score | Reading examination score |
| Writing Score | Writing examination score |

### 🎯 Target Variable

The target variable is `average_score`, calculated as:

```python
average_score = (
    math_score + reading_score + writing_score
) / 3
```

The ANN predicts this average score.

**Note:** Subject examination scores are used to calculate the target, not as input features. This avoids target leakage.

---

## 🧠 ANN Model Architecture

The project uses a feedforward Artificial Neural Network for a regression task.

### Model Structure

```text
Input Layer
    |
    | 17 Features
    |
Dense Layer (64 Neurons, ReLU)
    |
Dense Layer (32 Neurons, ReLU)
    |
Output Layer (1 Neuron, Linear)
    |
Predicted Average Score
```

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model | Sequential ANN |
| Hidden Layers | 2 |
| First Hidden Layer | 64 neurons |
| Second Hidden Layer | 32 neurons |
| Activation | ReLU |
| Output Neurons | 1 |
| Output Activation | Linear |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Evaluation Metric | Mean Absolute Error (MAE) |
| Early Stopping | Used during training |

---

## 📁 Project Structure

```text
Student-Performance-Prediction-ANN/
│
├── dataset/
│   └── StudentsPerformance.csv
│
├── notebooks/
│   └── student_performance.ipynb
│
├── models/
│   ├── ann_model_compatible.keras
│   └── preprocessor.pkl
│
├── app.py
├── fix_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/aakanshatomar3906-source/Student-Performance-Prediction-ANN.git
```

### 2. Navigate to the Project Directory

```bash
cd Student-Performance-Prediction-ANN
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 💻 How to Use the Application

1. Open the Student Performance Predictor application.
2. Select the student's gender.
3. Choose the race/ethnicity group.
4. Select parental education level.
5. Choose the lunch type.
6. Select the test preparation status.
7. Click **Predict Student Performance**.
8. View the estimated average examination score.

---



## 📈 Future Improvements

- Add model evaluation metrics and performance visualizations.
- Compare ANN performance with traditional Machine Learning algorithms.
- Improve model generalization through hyperparameter tuning.
- Add student performance category predictions.
- Deploy the application online.
- Develop a dashboard for exploring student performance patterns.

---

## 🎯 Learning Outcomes

Through this project, I explored:

- Building an Artificial Neural Network using Keras.
- Understanding regression problems in Deep Learning.
- Data preprocessing and categorical feature encoding.
- Training, validation, and early stopping.
- Saving and loading trained Deep Learning models.
- Integrating a trained ANN model with Streamlit.
- Building an interactive end-to-end AI application.

---

## ⚠️ Disclaimer

This project is developed for educational and learning purposes.

Predictions are estimates based on patterns learned from the provided dataset. They should not be considered actual examination results or used as the sole basis for academic decisions.

---

## 👩‍💻 Author

**Aakansha Tomar**

B.Tech – Artificial Intelligence & Data Science

- LinkedIn: [Aakansha Tomar](https://www.linkedin.com/in/aakanshatomar/)
- GitHub: [aakanshatomar3906-source](https://github.com/aakanshatomar3906-source)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Deep Learning, and curiosity.**
