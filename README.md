# 📩 SMS Spam Detection Using Machine Learning & Deep Learning

### NLP Text Classification | Naive Bayes | Logistic Regression | LSTM | Streamlit

**Innomatics Research Labs — Data Science & Deep Learning Project**

An end-to-end Natural Language Processing (NLP) project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using traditional machine learning and deep learning techniques.

The project compares **Multinomial Naive Bayes, Logistic Regression, and Long Short-Term Memory (LSTM)** models and implements an interactive Streamlit application for real-time SMS classification.

## 🚀 Live Application

Try the interactive SMS Spam Detection app:

**Streamlit Live Demo:** https://sms-spam-detector-lstm.streamlit.app/

Enter an SMS message to classify it as **Spam or Ham (Legitimate)** using the trained LSTM model.

**Test Performance:**
- Accuracy: 97.67%
- Precision: 94.07%
- Recall: 86.72%
- F1-Score: 90.24%

## 📌 Project Overview

Spam messages are an everyday communication challenge. They may contain misleading advertisements, promotional content, phishing attempts, or fraudulent offers.

This project develops an automated SMS classification system capable of identifying potentially unwanted messages.

Three approaches were implemented:

1. **Multinomial Naive Bayes** using TF-IDF features.
2. **Logistic Regression** using TF-IDF features.
3. **LSTM Neural Network** using learned word embeddings.

Models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

The LSTM model was selected for the interactive demonstration based on its highest test F1-score.

**Important:** The model detects patterns associated with SMS spam. It is not a comprehensive scam or phishing verification system.

---

## 🎯 Project Objectives

- Perform data cleaning and exploratory data analysis.
- Study the differences between spam and legitimate messages.
- Extract meaningful features from SMS text.
- Train Naive Bayes and Logistic Regression classifiers.
- Develop an LSTM-based deep learning classifier.
- Compare models using multiple evaluation metrics.
- Build an interactive Streamlit application.
- Save and reload the trained model for inference.

## 📚 Dataset Information

**Dataset:** SMS Spam Collection  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
**DOI:** 10.24432/C5CC84

The SMS Spam Collection dataset contains labeled English SMS messages classified as:

- **Ham (0):** Legitimate SMS.
- **Spam (1):** Unwanted or unsolicited SMS.

### Original Dataset Statistics

| Parameter | Value |
|---|---:|
| Total messages | 5,572 |
| Ham messages | 4,825 |
| Spam messages | 747 |
| Missing values | 0 |
| Duplicates removed during final preprocessing | 415 |
| Cleaned messages | 5,157 |

### Cleaned Dataset

| Category | Messages |
|---|---:|
| Ham | 4,516 |
| Spam | 641 |
| **Total** | **5,157** |

The dataset is imbalanced, with significantly more legitimate messages than spam messages.

This makes precision, recall, and F1-score important evaluation metrics beyond accuracy alone.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| NLTK | Text preprocessing |
| Scikit-learn | Machine learning and evaluation |
| TensorFlow / Keras | Deep learning |
| LSTM | Sequence-based text classification |
| TF-IDF | Text feature extraction |
| Streamlit | Interactive web application |
| Google Colab | Model development and training |
| GitHub | Source code hosting |

---

## 🔄 Project Workflow

```text
SMS Spam Collection Dataset
            |
            v
     Data Exploration
            |
            v
     Data Cleaning
            |
            v
  Feature Engineering
            |
            v
 Exploratory Data Analysis
            |
            v
  Stratified Train/Test Split
            |
            +---------------------------+
            |                           |
            v                           v
       TF-IDF Features              Tokenization
            |                           |
     +------+------+                    v
     |             |                Padding
     v             v                    |
 Naive Bayes   Logistic Regression      v
     |             |               Embedding
     |             |                    |
     |             |                    v
     |             |                   LSTM
     |             |                    |
     +------+------+--------------------+
            |
            v
     Model Evaluation
            |
            v
     Model Comparison
            |
            v
    Selected LSTM Model
            |
            v
    Streamlit Application
            |
            v
     SMS Classification
       Spam / Ham
```

## 🧹 Data Cleaning and Preprocessing

The following techniques were applied:

1. Verified dataset structure and data types.
2. Checked missing and blank messages.
3. Removed duplicate records.
4. Standardized class labels.
5. Cleaned and normalized SMS text.
6. Extracted text-based statistical features.
7. Converted class labels into numerical values.
8. Split the cleaned data using stratified sampling.

### Feature Engineering

Four text-based features were created:

- **Message Length:** Number of characters.
- **Word Count:** Number of words.
- **Punctuation Count:** Number of punctuation symbols.
- **Digit Count:** Number of numeric characters.

These features helped explore differences between spam and legitimate messages.

## 📊 Exploratory Data Analysis

The following analyses were performed:

1. Spam vs. Ham class distribution.
2. Message length distributions.
3. Word-frequency analysis for spam messages.
4. Word-frequency analysis for legitimate messages.
5. Spam word cloud.
6. Ham word cloud.
7. Feature correlation heatmap.

### Key Observations

- The dataset contains substantially more ham messages than spam messages.
- Text length and word-frequency patterns provide useful signals.
- Certain words and expressions occur more frequently in spam.
- Class imbalance requires careful evaluation.

## 🔀 Train/Test Split

The cleaned dataset was divided using an **80/20 stratified split** to preserve the class distribution.

| Dataset | Ham | Spam | Total |
|---|---:|---:|---:|
| Training | 3,612 | 513 | 4,125 |
| Testing | 904 | 128 | 1,032 |
| **Total** | **4,516** | **641** | **5,157** |

The same test split was used to compare the three approaches.

---

## 🤖 Model 1 — Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification. It estimates the probability of a message belonging to a class based on its text features.

### Configuration

- Feature extraction: TF-IDF
- Maximum TF-IDF features: 5,000
- Classifier: Multinomial Naive Bayes

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 96.90% |
| Precision | 100.00% |
| Recall | 75.00% |
| F1-score | 85.71% |
| ROC-AUC | 0.9809 |

**Confusion Matrix:**

```text
              Predicted
              Ham   Spam
Actual Ham    904     0
Actual Spam    32    96
```

### Observation

Naive Bayes produced no false positives on the test set but missed 32 actual spam messages.

Its high precision came at the cost of lower spam recall.

## 🤖 Model 2 — Logistic Regression

Logistic Regression is a supervised machine learning algorithm widely used for classification.

### Configuration

- Feature extraction: TF-IDF
- Maximum TF-IDF features: 5,000
- Classifier: Logistic Regression with class balancing

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 97.00% |
| Precision | 86.47% |
| Recall | 89.84% |
| F1-score | 88.12% |
| ROC-AUC | 0.9884 |

**Confusion Matrix:**

```text
              Predicted
              Ham   Spam
Actual Ham    886    18
Actual Spam    13   115
```

### Observation

Logistic Regression improved spam recall compared with Naive Bayes, reducing missed spam messages from 32 to 13.

However, it incorrectly classified 18 legitimate messages as spam.

---

## 🧠 Model 3 — LSTM Deep Learning Model

A Long Short-Term Memory (LSTM) neural network was developed to learn sequential relationships between words.

Unlike TF-IDF-based models, the LSTM processes token sequences and learns word representations through an embedding layer.

### Model Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 64 |
| LSTM units | 64 |
| Dropout rate | 0.30 |
| Maximum sequence length | 100 |
| Total parameters | 574,529 |
| Training strategy | EarlyStopping |
| Best epoch | 6 |
| Best validation loss | 0.0402 |

### Architecture

```text
SMS Message
     |
     v
Text Tokenization
     |
     v
Sequence Padding (100 Tokens)
     |
     v
Embedding Layer (64 Dimensions)
     |
     v
LSTM Layer (64 Units)
     |
     v
Dropout (0.30)
     |
     v
Sigmoid Classification
     |
     v
Spam Probability
     |
     v
Spam / Ham
```

### LSTM Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | **97.67%** |
| Precision | **94.07%** |
| Recall | **86.72%** |
| F1-score | **90.24%** |
| ROC-AUC | **0.9881** |

**Confusion Matrix:**

```text
              Predicted
              Ham   Spam
Actual Ham    897     7
Actual Spam    17   111
```

### Observation

The LSTM achieved the highest overall F1-score and accuracy among the three models.

It provided a strong balance between detecting spam and avoiding false positives, although Logistic Regression achieved higher spam recall.

---

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 96.90% | 100.00% | 75.00% | 85.71% | 0.9809 |
| Logistic Regression | 97.00% | 86.47% | **89.84%** | 88.12% | **0.9884** |
| **LSTM** | **97.67%** | 94.07% | 86.72% | **90.24%** | 0.9881 |

### Why LSTM Was Selected

The LSTM model achieved:

- Highest test accuracy: **97.67%**
- Highest test F1-score: **90.24%**
- Only 7 false positives out of 904 legitimate messages.
- Strong discrimination between spam and ham.

It was selected for the interactive demonstration based on its highest F1-score.

For applications prioritizing the detection of as much spam as possible, Logistic Regression remains a useful alternative because of its higher recall.

## 🌐 Streamlit Web Application

The trained LSTM model was integrated into an interactive Streamlit application for real-time SMS classification.

### Application Features

- Enter any SMS message.
- Predict whether the message is Spam or Ham.
- Display the classification result.
- Show model confidence.
- Reuse the saved LSTM model without retraining.
- Provide a simple interface suitable for live demonstrations.

### Example Messages

**Example 1 — Spam**

```text
Congratulations! You have won a free prize.
Click the link now to claim your reward.
```

**Example 2 — Ham**

```text
Hi, are we still meeting tomorrow at 10 AM?
```

**Example 3 — Potential Spam**

```text
URGENT: Your account will be blocked.
Verify your details immediately.
```

These are illustrative input examples, not a record of validated predictions for each message.

## 📂 Project Structure

```text
SMS-Spam-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── sms_spam_lstm.keras
├── sms_vectorizer_config.json
│
├── notebooks/
│   └── SMS_Spam_Ham_Classification.ipynb
│
└── screenshots/
    ├── class_distribution.png
    ├── message_length_distribution.png
    ├── spam_wordcloud.png
    ├── ham_wordcloud.png
    ├── model_comparison.png
    └── normal_HAM.png
    ├── OTP_HAM.png
    ├── prize_SPAM.png
    ├── promotional_SPAM.png
```

*The structure above is a suggested repository organization. Match the notebook and screenshot filenames to your actual saved files.*

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/suryaprabhag/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv .venv
```

macOS / Linux:

```bash
python3 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

**Note:** The repository's `requirements.txt` should specify compatible versions of TensorFlow, Streamlit, and other dependencies used by `app.py`.

## 💾 Saved Model Artifacts

The LSTM model and tokenizer configuration were saved for reuse.

| Artifact | Size |
|---|---:|
| `sms_spam_lstm.keras` | 6,924,744 bytes |
| `sms_vectorizer_config.json` | 122,917 bytes |

The saved model was reloaded and checked for prediction consistency.

## ⚠️ Limitations

- The dataset primarily contains English SMS messages.
- Spam language and messaging tactics change over time.
- Unfamiliar messages may be classified incorrectly.
- The dataset has a significant class imbalance.
- The model may miss sophisticated phishing or scam messages.
- High accuracy on this dataset does not guarantee equivalent real-world performance.
- Predictions should not be the sole basis for security decisions.

## 🔭 Future Improvements

1. Train using larger, more recent SMS datasets.
2. Compare LSTM with GRU and bidirectional LSTM.
3. Experiment with transformer-based classifiers.
4. Evaluate multilingual SMS messages.
5. Improve probability calibration.
6. Add explainability for model predictions.
7. Monitor classification drift using new labeled data.
8. Improve detection of previously unseen spam patterns.

## 🎓 Key Learning Outcomes

This project provided practical experience in:

- Data cleaning and validation.
- NLP text preprocessing.
- Exploratory Data Analysis.
- TF-IDF feature extraction.
- Naive Bayes and Logistic Regression.
- Word embeddings and LSTM networks.
- Imbalanced classification evaluation.
- Confusion matrices and ROC-AUC analysis.
- Deep learning model serialization.
- Interactive application development with Streamlit.

## 🏁 Conclusion

The project demonstrates a complete NLP classification pipeline for detecting spam and legitimate SMS messages.

Three classification models were evaluated on the same stratified test dataset.

The **LSTM achieved the highest test F1-score of 90.24% and accuracy of 97.67%**, while Logistic Regression delivered the highest spam recall at 89.84%.

The trained LSTM was integrated into a Streamlit application to demonstrate interactive SMS classification.

This project provides a practical foundation for more advanced text classification and spam detection systems.

---

## 👨‍💻 Author

**SuryaPrabha Rebba**

**GitHub:** https://github.com/suryaprabhag

**Project Repository:** https://github.com/suryaprabhag/SMS-Spam-Detection

**Program:** Data Science / Deep Learning / NLP — Innomatics Research Labs

## 🙏 Acknowledgements

Special thanks to **Innomatics Research Labs**, the mentors, and the training team for providing guidance and hands-on learning opportunities in Machine Learning, Natural Language Processing, and Deep Learning.
