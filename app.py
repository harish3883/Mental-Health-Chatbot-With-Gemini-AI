# import nltk
# nltk.download('popular')
# nltk.download('punkt_tab')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import pickle
# import numpy as np
# from keras.models import load_model
# model = load_model('model.h5')
# import json
# import random

# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# import spacy
# from spacy.language import Language
# from spacy_langdetect import LanguageDetector

# # Translator pipeline for English to Swahili translations
# eng_swa_tokenizer = AutoTokenizer.from_pretrained("C:/Users/noora/Downloads/drive-download-20240906T025307Z-001")
# eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/noora/Downloads/drive-download-20240906T025307Z-001")

# eng_swa_translator = pipeline(
#     "text2text-generation",
#     model=eng_swa_model,
#     tokenizer=eng_swa_tokenizer,
# )

# def translate_text_eng_swa(text):
#     translated_text = eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']
#     return translated_text

# # Translator pipeline for Swahili to English translations
# swa_eng_tokenizer = AutoTokenizer.from_pretrained("C:/Users/noora/Downloads/swa_eng_model-20240906T024457Z-001/swa_eng_model")
# swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/noora/Downloads/swa_eng_model-20240906T024457Z-001/swa_eng_model")

# swa_eng_translator = pipeline(
#     "text2text-generation",
#     model=swa_eng_model,
#     tokenizer=swa_eng_tokenizer,
# )

# def translate_text_swa_eng(text):
#     translated_text = swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']
#     return translated_text

# # Function to detect language using spaCy
# def get_lang_detector(nlp, name):
#     return LanguageDetector()

# nlp = spacy.load('en_core_web_sm')
# Language.factory("language_detector", func=get_lang_detector)
# nlp.add_pipe('language_detector', last=True)

# # Load intents and model data
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('texts.pkl', 'rb'))
# classes = pickle.load(open('labels.pkl', 'rb'))

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)  
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s: 
#                 bag[i] = 1
#                 if show_details:
#                     print(f"found in bag: {w}")
#     return np.array(bag)

# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# def getResponse(ints, intents_json):
#     if ints:
#         tag = ints[0]['intent']
#         for i in intents_json['intents']:
#             if i['tag'] == tag:
#                 return random.choice(i['responses'])
#     return "Sorry, I didn't understand that."

# def chatbot_response(msg):
#     doc = nlp(msg)
#     detected_language = doc._.language['language']
#     print(f"Detected language in chatbot_response: {detected_language}")
    
#     chatbotResponse = "Loading bot response..........."
    
#     # If the language is English, proceed as usual
#     if detected_language == "en":
#         res = getResponse(predict_class(msg, model), intents)
#         chatbotResponse = res
#         print("Chatbot response in English:", res)
        
#     # If the language is Swahili, translate to English, process, then translate back
#     elif detected_language == 'sw':
#         translated_msg = translate_text_swa_eng(msg)
#         print(f"Translated Swahili message to English: {translated_msg}")
#         res = getResponse(predict_class(translated_msg, model), intents)
#         chatbotResponse = translate_text_eng_swa(res)
#         print("Chatbot response in Swahili:", chatbotResponse)

#     return chatbotResponse

# from flask import Flask, render_template, request
# app = Flask(__name__)
# app.static_folder = 'static'

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     print(f"User message: {userText}")

#     doc = nlp(userText)
#     detected_language = doc._.language['language']
#     print(f"Detected language in get_bot_response: {detected_language}")

#     bot_response_translate = "Loading bot response..........."
    
#     if detected_language == "en":
#         bot_response_translate = userText  
#         print("English message:", bot_response_translate)
        
#     elif detected_language == 'sw':
#         bot_response_translate = translate_text_swa_eng(userText)
#         print(f"Translated Swahili message to English: {bot_response_translate}")

#     chatbot_response_text = chatbot_response(bot_response_translate)

#     if detected_language == 'sw':
#         chatbot_response_text = translate_text_eng_swa(chatbot_response_text)
#         print(f"Translated chatbot response to Swahili: {chatbot_response_text}")

#     return chatbot_response_text

# if __name__ == "__main__":
#     app.run()
# import nltk
# nltk.download('popular')  # Download popular NLTK data for tokenization, etc.
# from nltk.stem import WordNetLemmatizer
# import numpy as np
# import json
# import random
# from keras.models import load_model
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from flask import Flask, render_template, request
# import pickle

# # Initialize Flask app
# app = Flask(__name__)
# app.static_folder = 'static'

# # Load model and data
# lemmatizer = WordNetLemmatizer()
# model = load_model('model.h5')
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('texts.pkl', 'rb'))
# classes = pickle.load(open('labels.pkl', 'rb'))

# # Placeholder for gemini_search API call
# def gemini_search(query):
#     url = "https://api.gemini.com/v1/answer"
#     headers = {
#         "A
#         "Content-Type": "application/json"
#     }
#     data = {
#         "prompt": prompt,
#         "max_tokens": 50  # Adjust based on the desired response length
#     }
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code == 200:
#         return response.json().get("answer")
#     else:
#         print("Error with Gemini API:", response.status_code)
#         return "I'm sorry, I couldn't find an answer to that. can you specify more"
# # Text processing functions
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             return result
#     return None

# def chatbot_response(msg):
#     ints = predict_class(msg, model)
#     if not ints:
#         # If no intent is found, fallback to Gemini API
#         return gemini_search(msg)
#     res = getResponse(ints, intents)
#     if not res:
#         return gemini_search(msg)
#     return res

# # Updated intents.json for dynamic responses (Example format)
# intents = {
#     "intents": [
#         {
#             "tag": "greeting",
#             "patterns": ["Hi", "Hello", "How are you?", "Is anyone there?", "Hey", "Hola", "Hi there"],
#             "responses": ["Hello, how can I help you today?", "Hi there, I'm here to help you. What's on your mind?"]
#         },
#         {
#             "tag": "mood_improvement",
#             "patterns": ["I'm feeling down", "How can I improve my mood?", "Make me feel better"],
#             "responses": [
#                 "I'm sorry you're feeling this way. Sometimes a short walk or a favorite song can help. Want to try that?",
#                 "Taking deep breaths and talking it out can make a difference. Would you like to talk more about what's bothering you?"
#             ]
#         },
#         {
#             "tag": "feeling_bad",
#             "patterns": ["I feel bad", "I'm not feeling well", "I'm down", "I had a rough day"],
#             "responses": [
#                 "I'm really sorry to hear that. Want to tell me more about it?",
#                 "That sounds tough. I'm here to listen if you want to talk."
#             ]
#         },
#         {
#             "tag": "anxiety",
#             "patterns": ["I'm anxious", "I feel nervous", "I'm stressed"],
#             "responses": [
#                 "Anxiety can be challenging. Have you tried deep breathing or grounding exercises?",
#                 "I'm here for you. Sometimes just talking about what's on your mind can help with anxiety."
#             ]
#         },
#         {
#             "tag": "goodbye",
#             "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
#             "responses": ["Goodbye! Take care and remember, I'm here if you need to talk again.", "See you later!"]
#         }
#     ]
# }

# # Flask routes for frontend
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     print(f"User Input: {userText}")
#     response = chatbot_response(userText)
#     print(f"Bot Response: {response}")
#     return response

# # Start the Flask app
# if __name__ == "__main__":
#     app.run()
# uthorization": "AIzaSyCTGcjywcHSgafwh5U785-e646c-vrHyW0",
# import nltk
# from google.generativeai import generativeai

# nltk.download('popular')  # Download popular NLTK packages
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import pickle
# import numpy as np
# from keras.models import load_model
# import json
# import random
# from google.generativeai import GenerateText  # Import the Gemini model

# # Initialize the Gemini API model with your API key
# model = GenerateText(model="gemini-pro", api_key="AIzaSyCTGcjywcHSgafwh5U785-e646c-vrHyW0")

# # Load intents and pre-trained model
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('texts.pkl', 'rb'))
# classes = pickle.load(open('labels.pkl', 'rb'))
# model = load_model('model.h5')

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)  
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return np.array(bag)

# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
#     return return_list

# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result

# def chatbot_response(msg):
#     ints = predict_class(msg, model)
#     res = getResponse(ints, intents)
#     print(res)
    
#     return res

# def gemini_api_query(prompt):
#     # Query Gemini API when the prompt is not found in intents
#     response = model.generate(prompt=prompt)
#     return response.text

# # Flask setup for the chatbot
# from flask import Flask, render_template, request
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     print(userText)

#     # Check if user query is in intents, else query Gemini API
#     ints = predict_class(userText, model)
#     if ints:
#         res = getResponse(ints, intents)
#     else:
#         # If the prompt is not available in the intents, use Gemini API
#         prompt = f"The user is asking: {userText}. Answer in a helpful and informative way."
#         res = gemini_api_query(prompt)

#     return chatbot_response(userText)

# if __name__ == "__main__":
#     app.run(debug=True)
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
google.generativeai.configure(api_key="AIzaSyCTGcjywcHSgafwh5U785-e646c-vrHyW0")

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
