import streamlit as st

# MUST BE FIRST - Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter notebook first to train the models.")
        return None, None

# Text preprocessing function
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .ham {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📱 SMS Spam Detector")
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses Machine Learning to detect spam messages in SMS texts.
    
    ### Models Used:
    - Naive Bayes (Primary)
    - Logistic Regression
    - Support Vector Machine
    - Random Forest
    - Decision Tree
    
    ### Features:
    - Real-time spam detection
    - Confidence score
    - Batch processing
    - Model performance metrics
    """)
    st.markdown("---")
    st.markdown("### Instructions")
    st.info("""
    1. Enter a message in the text area
    2. Click 'Predict' to classify
    3. View the result and confidence
    """)

# Main content
st.title("📱 SMS Spam Detection System")
st.markdown("### Detect spam messages using Machine Learning")

# Load models
model, vectorizer = load_models()

if model is not None and vectorizer is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Processing", "📈 Model Performance", "ℹ️ About Project"])
    
    with tab1:
        st.header("Single Message Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input text area
            user_input = st.text_area(
                "Enter SMS message:",
                height=150,
                placeholder="Type your message here...",
                help="Enter the SMS message you want to classify"
            )
            
            # Predict button
            if st.button("🔍 Predict", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Analyzing message..."):
                        # Preprocess and predict
                        processed_text = preprocess_text(user_input)
                        vectorized_text = vectorizer.transform([processed_text])
                        prediction = model.predict(vectorized_text)[0]
                        probability = model.predict_proba(vectorized_text)[0]
                        
                        # Display result
                        if prediction == 1:
                            st.markdown(
                                '<div class="prediction-box spam">🚨 SPAM DETECTED</div>',
                                unsafe_allow_html=True
                            )
                            confidence = probability[1] * 100
                        else:
                            st.markdown(
                                '<div class="prediction-box ham">✅ LEGITIMATE MESSAGE (HAM)</div>',
                                unsafe_allow_html=True
                            )
                            confidence = probability[0] * 100
                        
                        # Confidence meter
                        st.metric("Confidence Level", f"{confidence:.2f}%")
                        st.progress(confidence / 100)
                        
                        # Additional info
                        with st.expander("📋 Message Details"):
                            st.write(f"**Original Message:** {user_input}")
                            st.write(f"**Processed Message:** {processed_text}")
                            st.write(f"**Message Length:** {len(user_input)} characters")
                            st.write(f"**Word Count:** {len(user_input.split())} words")
                            st.write(f"**Spam Probability:** {probability[1]:.4f}")
                            st.write(f"**Ham Probability:** {probability[0]:.4f}")
                else:
                    st.warning("Please enter a message to classify.")
        
        with col2:
            st.subheader("Sample Messages")
            st.markdown("**Try these examples:**")
            
            examples = [
                "Congratulations! You've won a $1000 gift card. Click here to claim.",
                "Hey, are we still on for dinner tonight?",
                "URGENT! Your bank account has been compromised. Call immediately.",
                "Can you pick up milk on your way home?",
                "FREE entry to a prize draw! Text WIN to 12345",
                "Meeting rescheduled to 3pm tomorrow"
            ]
            
            for idx, example in enumerate(examples, 1):
                if st.button(f"Example {idx}", key=f"example_{idx}", use_container_width=True):
                    st.session_state.example_text = example
            
            if 'example_text' in st.session_state:
                st.code(st.session_state.example_text, language=None)
    
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload a CSV file with messages to classify multiple SMS at once")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should have a column named 'message'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'message' in df.columns:
                    st.success(f"✅ File uploaded successfully! Found {len(df)} messages.")
                    
                    if st.button("🔍 Classify All Messages", type="primary"):
                        with st.spinner("Processing messages..."):
                            # Process all messages
                            processed_messages = df['message'].apply(preprocess_text)
                            vectorized_messages = vectorizer.transform(processed_messages)
                            predictions = model.predict(vectorized_messages)
                            probabilities = model.predict_proba(vectorized_messages)
                            
                            # Add results to dataframe
                            df['Prediction'] = ['Spam' if p == 1 else 'Ham' for p in predictions]
                            df['Confidence'] = [max(prob) * 100 for prob in probabilities]
                            df['Spam_Probability'] = [prob[1] for prob in probabilities]
                            
                            # Display results
                            st.subheader("Classification Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                spam_count = sum(predictions == 1)
                                st.metric("Spam Messages", spam_count)
                            with col2:
                                ham_count = sum(predictions == 0)
                                st.metric("Ham Messages", ham_count)
                            with col3:
                                spam_percentage = (spam_count / len(df)) * 100
                                st.metric("Spam Percentage", f"{spam_percentage:.1f}%")
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(8, 5))
                            df['Prediction'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
                            ax.set_title('Classification Results Distribution')
                            ax.set_xlabel('Category')
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="spam_detection_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("❌ CSV file must contain a 'message' column!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file to get started")
            st.markdown("""
            **CSV Format Example:**
            ```
            message
            "Congratulations! You won a prize"
            "Hey, how are you doing?"
            "Click here for free money"
            ```
            """)
    
    with tab3:
        st.header("Model Performance Metrics")
        
        # Performance data (from training)
        performance_data = {
            'Model': ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree'],
            'Accuracy': [0.9838, 0.9820, 0.9838, 0.9739, 0.9514],
            'Precision': [0.9722, 0.9859, 0.9861, 0.9773, 0.9032],
            'Recall': [0.9441, 0.9161, 0.9231, 0.8951, 0.8881],
            'F1-Score': [0.9579, 0.9496, 0.9533, 0.9343, 0.8956]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        st.subheader("📊 Comparative Model Performance")
        st.dataframe(perf_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), 
                     use_container_width=True)
        
        # Metrics visualization
        st.subheader("📈 Performance Comparison Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = range(len(perf_df))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                ax.bar([p + width * i for p in x], perf_df[metric], width, label=metric)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks([p + width * 1.5 for p in x])
            ax.set_xticklabels(perf_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            perf_df.set_index('Model')['Accuracy'].plot(kind='barh', ax=ax, color='skyblue')
            ax.set_xlabel('Accuracy Score')
            ax.set_title('Accuracy Comparison Across Models')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("🎯 Model Selection Rationale")
        st.info("""
        **Naive Bayes** was selected as the primary model because:
        - ✅ Highest accuracy (98.38%)
        - ✅ Excellent precision (97.22%)
        - ✅ Best recall among top performers (94.41%)
        - ✅ Fast training and prediction time
        - ✅ Works well with text classification tasks
        - ✅ Handles high-dimensional sparse data efficiently
        """)
    
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This SMS Spam Detection system uses Natural Language Processing (NLP) and Machine Learning 
        techniques to automatically classify SMS messages as either spam or legitimate (ham).
        
        ### 📊 Dataset Information
        - **Source:** UCI SMS Spam Collection Dataset
        - **Total Messages:** 5,574
        - **Spam Messages:** 747 (13.4%)
        - **Ham Messages:** 4,827 (86.6%)
        - **Features:** Text messages in English
        
        ### 🔧 Methodology
        
        **1. Data Preprocessing:**
        - Text lowercasing
        - Special character removal
        - Tokenization
        - Stopword removal
        - Stemming using Porter Stemmer
        
        **2. Feature Extraction:**
        - TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization
        - Maximum 3000 features
        
        **3. Models Implemented:**
        - Naive Bayes Classifier
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Random Forest Classifier
        - Decision Tree Classifier
        
        **4. Evaluation Metrics:**
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        
        ### 🏆 Key Results
        - Best Model: **Naive Bayes**
        - Accuracy: **98.38%**
        - Precision: **97.22%**
        - Recall: **94.41%**
        
        ### 💡 Use Cases
        - Mobile carrier spam filtering
        - Personal message filtering
        - Business communication security
        - Fraud detection
        
        ### 🔮 Future Enhancements
        - Deep Learning models (LSTM, BERT)
        - Multi-language support
        - Real-time SMS integration
        - User feedback mechanism
        - Explainable AI features
        
        ### 👨‍💻 Technical Stack
        - Python 3.x
        - Scikit-learn
        - NLTK
        - Pandas, NumPy
        - Streamlit
        - Matplotlib, Seaborn
        """)
        
        st.success("✅ Project completed as part of Machine Learning course assignment")

else:
    st.error("""
    ### ⚠️ Models Not Found
    Please run the Jupyter notebook (`sms_spam_detection.ipynb`) first to:
    1. Train the machine learning models
    2. Generate the pickle files (`naive_bayes_model.pkl` and `tfidf_vectorizer.pkl`)
    3. Then restart this Streamlit application
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>SMS Spam Detection System | Machine Learning Project</p>
    </div>
    """, unsafe_allow_html=True)