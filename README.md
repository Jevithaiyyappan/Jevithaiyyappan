# 📧 Fighting Spam with Python and Naive Bayes 
In today's digital age, spam has become a ubiquitous nuisance, flooding our inboxes and online platforms with unwanted messages. 
This project tackles the challenge head-on, providing a practical solution for identifying and filtering spam using the power of Python and machine learning.
This repository houses a simple yet effective spam detection model that leverages Natural Language Processing (NLP) techniques and the Naive Bayes algorithm. By analyzing the textual content of messages, our model learns to distinguish between spam (unsolicited, irrelevant messages) and ham (legitimate messages).

## ✨ Features

* **Data Preprocessing:** Cleans and prepares text data for analysis by removing stop words (common words like "the," "a," "is," etc.) and converting text to lowercase.
* **TF-IDF Vectorization:** Transforms text messages into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency), a technique that captures the importance of words in the context of spam.
* **Naive Bayes Classification:**  Employs the Naive Bayes algorithm, a probabilistic classifier known for its simplicity and effectiveness in text classification tasks.
* **Model Evaluation:** Assesses the model's performance using metrics like accuracy, confusion matrix, and classification report, providing insights into its strengths and areas for improvement.
This project demonstrates a simple yet effective spam detection model built with Python. It uses the power of Natural Language Processing (NLP) and the Naive Bayes algorithm to classify messages as spam or ham (not spam).


## 🚀 Getting Started

### 1. Prerequisites

* **Python:** Make sure you have Python installed on your system.
* **Libraries:** Install the required libraries using pip:
   ```bash
   pip install pandas scikit-learn nltk
2. Dataset
Download: You can use your own spam dataset or find one online (e.g., Kaggle, UCI Machine Learning Repository).
Placement: Place your dataset file (spam_data.csv) in the same directory as the Python script.
Format: Ensure your dataset has at least two columns: one for the message text and one for the label ("spam" or "ham").
3. Running the Code
Clone the Repository:
git clone https://github.com/your-username/your-repository-name.git
Navigate to Directory:
cd your-repository-name
Execute:
python spam_detector.py 
📊 Results
After running the script, you'll see the following evaluation metrics:

Accuracy: The overall accuracy of the model in classifying spam and ham.
Confusion Matrix: A table showing the number of true positives, true negatives, false positives, and false negatives.
Classification Report: Provides precision, recall, F1-score, and support for each class (spam and ham).
💡 Next Steps
Experiment with Different Datasets: Try using larger or more diverse datasets to improve the model's robustness.
Explore Other Algorithms: Investigate other machine learning algorithms like Support Vector Machines (SVM) or deep learning models for potentially better performance.
Hyperparameter Tuning: Fine-tune the model's parameters to optimize its performance for your specific dataset.
Build a User Interface: Create a web application or command-line interface to make the spam detector more user-friendly.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


Source Code 

# 1. Project Setup and Data Acquisition
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# 2. Data Preparation
# --- Replace 'your_spam_dataset.csv' with the actual path to your dataset ---
data = pd.read_csv('C:\\Users\\Verticurl-User\\Desktop\\Spam mail.csv') 

print(data.head()) 
print(data.info())  

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['cleaned_text'] = data['Masseges'].apply(clean_text) 

X = data['cleaned_text']
y = data['Category'] 

# 3. Feature Engineering
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 4. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


Please find the Screenshot of my output 
![image](https://github.com/user-attachments/assets/71a64bd9-302b-49c8-9c22-84f52f615d53)


