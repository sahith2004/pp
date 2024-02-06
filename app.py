categories_data = {
    'financial_news': 'Latest news and updates related to finance and the financial markets like Stock market rallies as tech giants post strong earnings, Federal Reserve announces interest rate hike, Cryptocurrency prices surge amid market volatility, Investors react to the latest economic indicators, Global economic outlook for the next quarter',
    'other': 'General information not specific to finance',
    'stock_data': 'Real-time data and statistics about stocks and financial instruments',
    'portfolio_information': 'Details and insights about your investment portfolio',
}


import google.generativeai as genai
genai.configure(api_key="AIzaSyARxB1UsrHVl7IiJY0hyhTqiKnHK8qPCBg")

# Define a function to calculate the embeddings of the input text
def model_MM(input_text):
    categories_data_embeddings_MM_ = genai.embed_content(
        model="models/embedding-001",
        content=input_text,
        task_type="retrieval_document",
        title="Embedding of inputs")
    return categories_data_embeddings_MM_['embedding']

# calculate embedding of sentence2 and categories_data
categories_data_embeddings_MM_ = model_MM(list(categories_data.values()))


from sklearn.metrics.pairwise import cosine_similarity
def find_best_result(user_prompt, embedding_model_MM, categories_data_embeddings_MM):
    # Convert user prompt to embedding vector
    user_prompt_embedding_MM = embedding_model_MM([user_prompt])


    # Calculate cosine similarity with categories data embeddings
    similarity_scores_MM = cosine_similarity(user_prompt_embedding_MM, categories_data_embeddings_MM)
   

    # Find the index of the best result with the highest score for both models
    best_result_index_MM = np.argmax(similarity_scores_MM)
   

    # Get the best first result with the highest score for both models
    best_result_MM = list(categories_data.keys())[best_result_index_MM]
   

    return best_result_MM

import numpy as np
# User prompt
user_prompt = "what is the rating of my portfolio"

# Find the best result with the highest score for both models
best_result_MM = find_best_result(user_prompt, model_MM, categories_data_embeddings_MM_)

# Print the results
print("Best result from MM model:", best_result_MM)

from flask import Flask
from pyngrok import ngrok
from flask import Flask, request,render_template
import requests
portno = 8000
from flask import jsonify
from flask_cors import CORS
from flask import Flask, request,render_template
import requests
from flask import Flask, send_file




app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
  return render_template("chat.html")
@app.route('/prompt/<input_text>', methods=['GET'])
def get_response(input_text):
    # Call your query_engine function with the input_text
    bot_response = find_best_result(input_text, model_MM, categories_data_embeddings_MM_)

    # Prepare the response data
    response_data = {
        'messages': [
            {'content': input_text}
        ],
        'candidates': [
            {'content': bot_response}
        ]
    }

    # Return the response as JSON
    return response_data
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    bot_response = find_best_result(input, model_MM, categories_data_embeddings_MM_)

    return bot_response



