

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


import os

# Set your Google API key as an environment variable
os.environ['GOOGLE_API_KEY'] = 'AIzaSyARxB1UsrHVl7IiJY0hyhTqiKnHK8qPCBg'
genai.configure(api_key='AIzaSyARxB1UsrHVl7IiJY0hyhTqiKnHK8qPCBg')

with open('converted_text.txt', 'r') as text_file:
    # Read the content of the file
    text_content = text_file.read()
    
with open('terms.txt', 'r') as text_file:
    # Read the content of the file
    f_text_content = text_file.read()
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response
    
raw_text = text_content + f_text_content
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)



from flask import Flask
from pyngrok import ngrok
from flask import Flask, request,render_template
import requests
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

portno = 8000


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
  return render_template("chat.html")
@app.route('/prompt/<input_text>', methods=['GET'])
def get_response(input_text):
    # Call your query_engine function with the input_text
    bot_response = user_input(input_text)['output_text']
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
    bot_response = user_input(input)['output_text']

    return bot_response



