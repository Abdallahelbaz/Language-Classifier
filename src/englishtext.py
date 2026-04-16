import pandas as pd
import nltk
from nltk import word_tokenize, WordNetLemmatizer , pos_tag , word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
nltk.download("stopwords")
import pickle
import os

class EnglishText:

    def __init__(self):
        self.model=''
        self.vectorizer=''
        self.le=''
        self.load_model()
        self.load_vectorizer()
        self.load_LE()
        

    def load_model(self):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '..', 'models', 'model_english.pkl')
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)
    

    def load_vectorizer(self):
        base_dir= os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '..', 'models', 'vectorizer_english.pkl')
        with open(file_path, 'rb') as file:
            self.vectorizer = pickle.load(file)
    
    def load_LE(self):
        pass

    

    def classify_text(self, text):
        text=[text]
        cv = self.vectorizer
        new_text=cv.transform(text).toarray()
        return self.model.predict(new_text)