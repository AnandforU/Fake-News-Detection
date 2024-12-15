# Fake News Detection System 
This project is a Fake News Detection System built using machine learning techniques to classify news articles as either Real or Fake. 
It processes and analyzes the text content of news articles using natural language processing (NLP) methods and applies various machine learning models to achieve accurate classification.

## Features
- Dataset: Combines two datasets:
- True dataset: Contains real news articles.
- Fake dataset: Contains fake news articles.
## Preprocessing:
- Combines the title and text of articles into a single content field.
- Applies text cleaning, tokenization, and stemming.
- Removes stopwords to improve model performance.
- Machine Learning Models:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Performance Metrics:
- Computes accuracy on training and test datasets for each model.
## Installation and Setup
Prerequisites
- Python 3.x
- Required Libraries:
- pandas
- numpy
- nltk
- scikit-learn

## Clone the repository or download the project files.

Install the required libraries:
pip install pandas numpy scikit-learn nltk
Download NLTK stopwords:

import nltk
nltk.download('stopwords')
Place the datasets (true.csv and fake.csv) in the project directory.

## Code Description
Step-by-Step Process
1. Data Loading
The system reads the true.csv and fake.csv files using pandas.
2. Preprocessing
Drop unnecessary columns such as date.
Assign a Class label:
Real News: 0
Fake News: 1
Concatenate the title and text into a new column content.
Encode the subject column using LabelEncoder.
3. Text Cleaning
The content field undergoes cleaning:
Remove non-alphabetic characters.
Convert to lowercase.
Tokenize and remove stopwords.
Apply stemming using PorterStemmer.
4. Feature Extraction
Convert cleaned text into numerical representations using TfidfVectorizer.
5. Train-Test Split
The dataset is split into training and testing sets (80% train, 20% test) using train_test_split.
6. Model Training and Evaluation
Train three models:
Logistic Regression
Random Forest
Decision Tree
Evaluate each model using accuracy_score on both training and test datasets.
Display model performance metrics.
7. Prediction
Use the trained models to classify new or unseen news articles.

## Results
Model Performance
Model	Train Accuracy	Test Accuracy
Logistic Regression	99.19%	98.69%
Random Forest	100.0%	99.25%
Decision Tree	100.0%	99.55%

