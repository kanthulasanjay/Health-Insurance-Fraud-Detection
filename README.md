# 🏥 Health Insurance Claim Fraud Detection using ANN

An **Artificial Neural Network (ANN) based Health Insurance Claim Fraud Detection system** that predicts whether a health insurance claim is potentially fraudulent.

The project includes a trained deep learning model, preprocessing pipeline, feature engineering, optimized decision threshold, and an interactive **Streamlit web application** for real-time fraud prediction.

---

## 🚀 Project Overview

Healthcare insurance fraud can result in significant financial losses for insurance providers. This project uses machine learning and an Artificial Neural Network to identify potentially fraudulent insurance claims based on patient, provider, medical, financial, and claim-history information.

The Streamlit application allows users to enter claim details and receive:

* 🔢 Fraud probability
* 🚨 Fraudulent / Not Fraudulent prediction
* 📊 Visual probability indicator
* 📋 Submitted claim information

The deployed application uses the same feature engineering and preprocessing pipeline used during model training.

---

## ✨ Features

* 🤖 Artificial Neural Network based fraud detection
* 🏥 Health insurance claim analysis
* 📊 37 input features
* ⚙️ Automated feature engineering
* 🔄 Reusable preprocessing pipeline
* 🎯 Optimized classification threshold
* 🌐 Interactive Streamlit interface
* 📈 Fraud probability prediction
* 🚨 Automatic fraud classification
* 💾 Saved trained model for inference

---

## 📊 Model Performance

| Metric             |                     Value |
| ------------------ | ------------------------: |
| Model              | Artificial Neural Network |
| Input Features     |                        37 |
| Test Accuracy      |                **85.65%** |
| Decision Threshold |                  **0.54** |

The application displays these model details directly in the Streamlit interface.

The selected threshold is stored separately in `health_claims_best_threshold.json` and is currently set to **0.54**.

---

## 🧠 Feature Engineering

Before prediction, the application performs several feature-engineering operations.

### Date-Based Features

* Claim delay in days
* Days to policy expiration
* Claim year
* Claim month
* Claim day of week
* Service year
* Service month

### Claim-Based Features

* Claim amount per procedure
* Claim amount per stay day
* Procedures per stay day
* Claim-to-patient-cost ratio

### Risk Features

* Patient claim risk
* Provider claim risk
* Total previous claims

### Financial Features

* Patient out-of-pocket cost
* Deductible amount
* CoPay amount

These engineered features are created before the preprocessing and prediction stages.

---

## 🏗️ Project Architecture

```text
User Input
    │
    ▼
Streamlit Web Application
    │
    ▼
Input Validation
    │
    ▼
Feature Engineering
    │
    ▼
Saved Preprocessor
    │
    ▼
ANN Model
    │
    ▼
Fraud Probability
    │
    ▼
Decision Threshold (0.54)
    │
    ├── Probability > 0.54
    │       └── 🚨 FRAUDULENT CLAIM
    │
    └── Probability ≤ 0.54
            └── ✅ NOT FRAUDULENT
```

The application transforms the input using the saved preprocessing pipeline and then passes the processed data to the trained ANN model.

---

## 🖥️ Streamlit Application

The application provides separate sections for entering:

### 👤 Patient Information

* Patient ID
* Patient age
* Patient gender
* Patient city
* Patient state

### 📋 Claim Information

* Claim ID
* Claim amount
* Number of procedures
* Claim date
* Service date
* Policy expiration date

### 🏥 Provider Information

* Hospital ID
* Provider type
* Provider specialty
* Provider city
* Provider state

### 🩺 Medical Information

* Diagnosis code
* Procedure code
* Admission type
* Discharge type
* Service type

### 💰 Financial Information

* Length of stay
* Deductible amount
* CoPay amount

### 📊 Claim History

* Previous patient claims
* Previous provider claims
* Provider-patient distance
* Whether the claim was submitted late

These fields are implemented directly in the Streamlit application.

---

## 🔍 Prediction Process

When the user clicks **Predict Fraud**, the application:

1. Collects the entered claim information.
2. Creates a Pandas DataFrame.
3. Performs feature engineering.
4. Applies the saved preprocessing pipeline.
5. Converts the processed data into the required NumPy format.
6. Passes the data to the trained ANN model.
7. Generates a fraud probability.
8. Compares the probability against the `0.54` threshold.
9. Displays the final prediction.

The application displays both the fraud probability and decision threshold in the prediction result.

---

## 📁 Project Structure

```text
Health-Insurance-Claim-Fraud-Detection/
│
├── app.py
├── Health Insurance Claim Fraud Detection using ANN.ipynb
│
├── health_insurance_fraud_model.keras
├── health_claims_preprocessor.pkl
├── health_claims_best_threshold.json
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                                                     | Description                                    |
| -------------------------------------------------------- | ---------------------------------------------- |
| `app.py`                                                 | Streamlit web application                      |
| `Health Insurance Claim Fraud Detection using ANN.ipynb` | Model development and experimentation notebook |
| `health_insurance_fraud_model.keras`                     | Trained ANN model                              |
| `health_claims_preprocessor.pkl`                         | Saved preprocessing pipeline                   |
| `health_claims_best_threshold.json`                      | Saved classification threshold                 |
| `requirements.txt`                                       | Python dependencies                            |
| `README.md`                                              | Project documentation                          |

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Artificial Neural Networks
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle
* JSON
* Jupyter Notebook

The provided requirements include Streamlit, TensorFlow, Pandas, NumPy, Scikit-learn 1.6.1, and h5py.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kanthulasanjay/health-insurance-claim-fraud-detection.git
```

### 2. Navigate to the Project

```bash
cd health-insurance-claim-fraud-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Prediction Output

The application provides two possible outcomes.

### 🚨 Fraudulent Claim

If:

```text
Fraud Probability > 0.54
```

the application classifies the claim as:

```text
🚨 FRAUDULENT CLAIM
```

### ✅ Not Fraudulent

If:

```text
Fraud Probability ≤ 0.54
```

the application classifies the claim as:

```text
✅ NOT FRAUDULENT
```

The application also displays the fraud probability as a percentage and provides a probability progress bar.

---

## 🔐 Model & Preprocessing Files

The application expects the following files to be present in the project directory:

```text
health_insurance_fraud_model.keras
health_claims_preprocessor.pkl
health_claims_best_threshold.json
```

The application loads the Keras model, preprocessing object, and threshold when it starts.

---

## 📌 Important Note

This project is intended for **educational and demonstration purposes**.

The prediction produced by the model should not be treated as a definitive determination of insurance fraud. Real-world fraud investigation requires additional verification, domain expertise, and appropriate regulatory and organizational processes.

---

## 🔮 Future Improvements

* Add model explainability using SHAP or LIME
* Add fraud-risk explanations for individual predictions
* Add batch CSV prediction
* Add prediction history
* Add interactive analytics dashboard
* Improve model calibration
* Compare ANN with XGBoost, Random Forest, and other models
* Deploy the application to a cloud platform
* Add authentication and role-based access
* Add database integration for claim records

---

## 👨‍💻 Author

**Kanthula Sanjay**

Developed as a machine learning project for detecting potentially fraudulent health insurance claims using an Artificial Neural Network.


