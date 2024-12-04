import nltk
nltk.download('popular')  # Download popular NLTK packages
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import google.generativeai  # Corrected import

# Initialize the Gemini API model with your API key
google.generativeai.configure(api_key="REPLACE WITH YOUR API KEY")

# Load intents and pre-trained model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
model = load_model('model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print(res)
    
    return res

def gemini_api_query(prompt):
    # Query Gemini API when the prompt is not found in intents
    response = google.generativeai.generate_text(model="gemini-pro", prompt=prompt)
    return response.text

# Flask setup for the chatbot
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)

    # Check if user query is in intents, else query Gemini API
    ints = predict_class(userText, model)
    if ints:
        res = getResponse(ints, intents)
    else:
        # If the prompt is not available in the intents, use Gemini API
        prompt = f"The user is asking: {userText}. Answer in a helpful and informative way."
        res = gemini_api_query(prompt)

    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True)
