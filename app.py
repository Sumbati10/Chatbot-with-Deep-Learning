from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

app = Flask(__name__)

# Load your model and tokenizer
model = keras.models.load_model('model/chat_model.keras')  # Adjust path as needed

# Load the tokenizer
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('model/label_encoder.pickle', 'rb') as ecn_file:
    lbl_encoder = pickle.load(ecn_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

    # Get prediction from the model
    prediction = model.predict(padded_sequences)
    
    # Decode the prediction
    response_index = np.argmax(prediction)
    response = lbl_encoder.inverse_transform([response_index])[0]
    
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)

