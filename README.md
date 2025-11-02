# Ml-Project

📱SMS Spam Detection using Machine Learning

Project Overview

This project implements an intelligent SMS spam detection system using multiple machine learning algorithms
to classify text messages as either spam or legitimate (ham). The system addresses the critical problem of
unwanted and potentially harmful spam messages that plague mobile users worldwide, causing annoyance and
security risks.

Problem Statement

With the exponential growth of mobile communications, spam messages have become a significant issue
affecting millions of users. These unwanted messages not only cause inconvenience but may also contain
phishing attempts, fraudulent schemes, or malicious links. An automated spam detection system can help filter
these messages, protecting users from potential threats and improving their messaging experience.

Project Importance


User Safety: Protects users from phishing attacks and fraudulent schemes

Time Efficiency: Automatically filters unwanted messages, saving users time

Privacy Protection: Reduces exposure to unsolicited marketing and scams

Scalability: Can process thousands of messages instantly

Cost Reduction: Helps mobile carriers reduce spam-related complaints and support costs

Results Summary

Our implementation achieved exceptional performance across multiple machine learning models, with the
Naive Bayes classifier emerging as the best performer with 98.38% accuracy. The system successfully
distinguishes between spam and legitimate messages with high precision and recall, making it suitable for realworld deployment

📊 Dataset Information

Dataset Source

Name: UCI SMS Spam Collection Dataset

Source: Kaggle - SMS Spam Collection Dataset

Original Source: UCI Machine Learning Repository

Dataset Characteristics

Total Messages: 5,574

Spam Messages: 747 (13.4%)

Ham Messages: 4,827 (86.6%)

Language: English

Format: CSV file with labeled text messages

Data Preprocessing Steps

Our preprocessing pipeline transformed raw text messages into clean, analyzable data:

1. Data Cleaning:
   
Removed duplicate messages (403 duplicates found)
Handled missing values (none found)
Retained 5,171 unique messages for analysis

2. Text Preprocessing:

Lowercasing: Converted all text to lowercase for uniformity

Special Character Removal: Eliminated punctuation, numbers, and symbols using regex patterns

Tokenization: Split messages into individual words

Stopword Removal: Removed common English words that don't carry significant meaning

Stemming: Applied Porter Stemmer to reduce words to their root form

3. Feature Engineering:
   
Created message length features for exploratory analysis

Label encoding: Converted 'ham' to 0 and 'spam' to 1
Generated TF-IDF features capturing term importance
Data Distribution Analysis
The dataset exhibits class imbalance with legitimate messages dominating:
Ham messages comprise 86.6% of the dataset
Spam messages represent 13.4% of the dataset
This imbalance is realistic and reflects actual message distribution


Statistical analysis revealed:
Spam messages tend to be longer than ham messages
Spam contains more promotional words and urgent language
Ham messages use conversational and informal language

🔬 Methodology

Approach Overview

Our methodology follows a systematic machine learning pipeline designed for text classification tasks:

Raw Data → Preprocessing → Feature Extraction → Model Training → Evaluation → Deployment

Why This Approach?

1. Text Preprocessing:
    Essential for NLP tasks as raw text contains noise that can confuse models. Our
stemming and stopword removal focus the model on meaningful content words.

2. TF-IDF Vectorization: Chosen over simple word counts because:
   
Captures word importance relative to the entire corpus
Reduces impact of frequently occurring but less informative words
Creates numerical features suitable for ML algorithms
Handles sparse, high-dimensional text data efficiently

3. Multiple Model Comparison: Implemented five diverse algorithms to:
   
Identify the best performer for this specific task
Understand which model characteristics work best with text data
Provide robustness through ensemble possibilities
Alternative Approaches Considered

1. Deep Learning Models:
   
Considered: LSTM, GRU, BERT transformers

Decision: Not implemented due to:

Smaller dataset size (better suited for traditional ML)
Higher computational requirements
Longer training times
Traditional ML achieved excellent results (98%+ accuracy)

2. Word2Vec/GloVe Embeddings:
   
Considered: Pre-trained word embeddings

Decision: TF-IDF chosen because:
Simpler and more interpretable
Faster training and inference
Performs well on smaller datasets
Lower memory footprint

3. Ensemble Methods:
   
Considered: Stacking, Voting Classifiers
Decision: Single models sufficient as individual models achieved high accuracy

 width="908" height="283" alt="Screenshot 2025-11-02 151032" src="https://github.com/user-attachments/assets/daba7691-bce5-4262-a4a3-2d9f03487569" />

