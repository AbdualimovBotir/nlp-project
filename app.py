import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# TfidfVectorizer'ni yuklash yoki yaratish
try:
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("TfidfVectorizer fayli topilmadi. Yaratilyapti...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1, 1))
    # Trening uchun namunaviy ma'lumotlar
    training_data = ["This is a sample review", "Another review here"]
    tfidf_vectorizer.fit(training_data)
    pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    print("TfidfVectorizer saqlandi!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Foydalanuvchidan kirishlarni olish
    star = request.form['star']
    review_text = request.form['text']

    # Preprocessing
    ps = PorterStemmer()
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)
    review_text = review_text.lower().split()
    review_text = [ps.stem(word) for word in review_text if not word in stopwords.words('english')]
    review_text = ' '.join(review_text)

    # TF-IDF vektorizatsiya
    review_features = tfidf_vectorizer.transform([review_text]).toarray()

    # Xususiyatlar va starni birlashtirish
    features = np.hstack([review_features, np.array([[int(star)]]).reshape(1, -1)])

    # Modelni prediktsiya qilish
    prediction = model.predict(features)

    # Natijani yuborish
    return render_template('index.html', prediction_text=f'Predicted Rating: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
