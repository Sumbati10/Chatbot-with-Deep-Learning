from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow import keras
import pickle

app = Flask(__name__)

# Load trained model and tokenizer
model = keras.models.load_model('chat_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

with open("intents.json") as file:
    data = json.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chatbot response route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')  # Extract user input from JSON
    
    # Process the input
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), 
                                             truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    # Find the response corresponding to the tag
    for intent in data['intents']:
        if intent['tag'] == tag:
            return jsonify({'response': np.random.choice(intent['responses'])})

    return jsonify({'response': "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True)

