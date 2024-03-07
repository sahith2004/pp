import transformers
import torch
from torch import cuda,bfloat16

model_id = 'PlusCash/TinyLlama-pluscash-finance-chat-v0.1'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.4,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=128,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

prompt = "How mutual funds works?"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest financial assistant.Your name is Finark personal assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Only answer the questions which are related to finance never ever answer questions regarding entertainement,sports,...etc which are out of finance domain.Also Only answer about the Indian stock market information and Indian mutual funds and tax information.
Only answer to the Questions with Keeping Indian context and Indian financial markets as context.
<</SYS>>
{prompt}[/INST]'''

from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

from transformers import pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

import locale
locale.getpreferredencoding = lambda: "UTF-8"

from flask import Flask
from pyngrok import ngrok

portno = 8000

from flask import jsonify
from flask_cors import CORS

import requests
from flask import Flask, request,render_template
app = Flask(__name__)
CORS(app)

ngrok.set_auth_token('2ZGr4Gf5m7wNAUvDwHdgdSSZJPX_4mGbGzSCpfAFuc7fPpGGi')
public_url = ngrok.connect(portno).public_url

@app.route("/")
def home():
  return render_template("chat.html")
@app.route('/prompt/<input_text>', methods=['GET'])
def get_response(input_text):
    # Call your query_engine function with the input_text
    prompt_template=f'''[INST] <<SYS>>
     You are a helpful, respectful and honest financial assistant.Your name is Finark personal assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Only answer the questions which are related to finance never ever answer questions regarding entertainement,sports,...etc which are out of finance domain.Always answer with Indian context.Be careful If somebody asks about any question try to answer only with respect to Indian markets and Indian financial rules and regulations.
     <</SYS>>
     {input_text}[/INST]'''
    print(input_text)
    result = summarizer(pipe(prompt_template)[0]['generated_text'][len(prompt_template):], max_length=500, min_length=30, do_sample=False)

    bot_response = result[0]['summary_text']
    print(bot_response)
    # Prepare the response data
    response_data = {
        'messages': [
            {'content': input_text}
        ],
        'candidates': [
            {'content': bot_response+"I am a finark assistant I "}
        ]
    }

    # Return the response as JSON
    return response_data
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    prompt_template=f'''[INST] <<SYS>>
     You are a helpful, respectful and honest financial assistant.Your name is Finark personal assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
     <</SYS>>
     {input}[/INST]'''
    print(input)
    result = summarizer(pipe(prompt_template)[0]['generated_text'][len(prompt_template):], max_length=100, min_length=30, do_sample=False)
    bot_response = result[0]['summary_text']
    print(bot_response)
    return bot_response



print(f"to access go to {public_url}")
app.run(port=portno)


