# Chatbot-with-Deep-Learning

[![Demo Video](https://github.com/Sumbati10/Chatbot-with-Deep-Learning/blob/main/Recording%202024-10-24%20205542.gif)

## Chatbot Deep Learning Interface

This project is a simple web-based chatbot powered by deep learning, designed to interact with users and provide responses based on user inputs. The backend is built with Flask, and the deep learning model is implemented using TensorFlow and Keras.

## Features

- User-friendly interface for chatting with an AI chatbot.
- Deep learning model that understands user inputs and responds accordingly.
- AJAX-based chat interaction for a seamless user experience.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework in Python.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **HTML/CSS**: For creating a responsive and attractive user interface.
- **JavaScript**: For enabling AJAX functionality in the chat interface.

## Training the Model

The deep learning model used in this chatbot was trained on a dataset that includes various user intents and corresponding responses. The training process involves the following steps:

1. **Data Collection**: Gather user intents and responses, formatted in a JSON file (`intents.json`).
2. **Preprocessing**: Use a tokenizer to convert text data into sequences that can be fed into the model.
3. **Model Training**: Build and train a neural network using TensorFlow/Keras with the preprocessed data.
4. **Model Saving**: After training, the model is saved as `chatbot_model.keras`, which is used in the Flask application for generating responses.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Pip (Python package manager)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sumbati10/Chatbot-with-Deep-Learning.git
   cd Chatbot-with-Deep-Learning
   
2. **Running the Application**
To start the Flask application, run the following command in your terminal:


python.py



The application will start on http://127.0.0.1:5000/. Open this URL in your web browser to access the chatbot interface.

<img width="946" alt="image" src="https://github.com/user-attachments/assets/08a1964e-91f9-4e24-82e4-ad76f16212f4">

# Usage
Type your message in the input field and click "Send" or press Enter.
The chatbot will respond based on the trained deep learning model.
You can have a continuous conversation with the chatbot.

# File Descriptions
app.py: Contains the main Flask application code. It handles user requests, processes input through the deep learning model, and returns responses.

index.html: The main HTML file that structures the chatbot interface.

style.css: CSS file that styles the chatbot interface, making it visually appealing and user-friendly.

chatbot_model.keras: The trained Keras model for processing and understanding user input.

tokenizer.pickle: Tokenizer object used for converting user input into a format suitable for the model.

label_encoder.pickle: Label encoder object used to decode model predictions into human-readable responses.

intents.json: A JSON file that defines various intents and their corresponding responses.

