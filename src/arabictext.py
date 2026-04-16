import pickle
import pickle
import os

class ArabicText:

    def __init__(self):
        self.model=''
        self.vectorizer=''
        self.le=''
        self.load_model()
        self.load_vectorizer()
        self.load_LE()
        

    def load_model(self):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '..', 'models', 'model_arabic.pkl')
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)
    

    def load_vectorizer(self):
        base_dir= os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '..', 'models', 'vectorizer_arabic.pkl')
        with open(file_path, 'rb') as file:
            self.vectorizer = pickle.load(file)
    
    def load_LE(self):
        base_dir= os.path.dirname(__file__)
        file_path = os.path.join(base_dir, '..', 'models', 'LE_arabic.pkl')
        with open(file_path, 'rb') as file:
            self.le = pickle.load(file)

    

    def classify_text(self, text):
        text=[text]
        cv = self.vectorizer
        new_text=cv.transform(text).toarray()
        return self.le.inverse_transform(self.model.predict(new_text))
        # return self.model.predict(new_text)