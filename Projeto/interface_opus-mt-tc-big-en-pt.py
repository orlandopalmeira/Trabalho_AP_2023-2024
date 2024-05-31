import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Miguelcj1/model_opus-mt-tc-big-en-pt-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

MAX_LENGTH = 128

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)        
    translated = model.generate(**inputs, max_length=MAX_LENGTH, num_return_sequences=1)
    decoded_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return decoded_translation[0]

def chatbot(message):
    translated_message = translate_text(">>por<< " + message)
    return translated_message

demo_chatbot = gr.Interface(chatbot, "textbox", "text", title="Translation Chatbot (opus-mt-tc-big-en-pt)", description="Enter text in English.")
demo_chatbot.launch()