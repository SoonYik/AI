Satirical Sentiment Analysis - Setup Instructions

1. PIP INSTALL:
pip install pandas numpy scikit-learn nltk joblib streamlit torch transformers scipy

2. NLTK DATA SET UP :
To avoid the Windows "file lock" or "not a zip file" error, run this command in your terminal BEFORE starting the app to safely download the required text processing data:
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

3. FILE STRUCTURE:
Ensure all these files are sitting together in the exact same folder:
- GUI_presentation.py
- svm_sentiment_model.pkl
- svm_sarcasm_model.pkl
- nb_sentiment_model.pkl
- nb_sarcasm_model.pkl
- /checkpoints/ (BERT models)

4. RUN THE APP:
Open your terminal, navigate to the folder containing your files, and execute:
python -m streamlit run GUI_presentation.py