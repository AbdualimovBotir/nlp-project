import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model and vectorizer from pickle files
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Load your trained vectorizer

# Pre-load stopwords and stemmer for better performance
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get inputs from the user
    star = request.form['star']
    review_text = request.form['text']

    # Preprocessing: clean the review text
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)  # Remove non-alphabetic characters
    review_text = review_text.lower().split()  # Convert to lower case and split into words
    review_text = [ps.stem(word) for word in review_text if word not in stop_words]  # Remove stopwords and apply stemming
    review_text = ' '.join(review_text)  # Join the cleaned words back into a string

    # TF-IDF Vectorization: Use the pre-fitted vectorizer
    review_features = tfidf_vectorizer.transform([review_text]).toarray()  # Use transform, not fit_transform

    # Combine features (TF-IDF features + star rating)
    features = np.hstack([review_features, np.array([[int(star)]]).reshape(1, -1)])

    # Make prediction using the model
    prediction = model.predict(features)

    # Send the prediction result back to the HTML template
    return render_template('index.html', prediction_text=f'Predicted Rating: {prediction[0]}')


if __name__ == "__main__":
    app.run(debug=True)
