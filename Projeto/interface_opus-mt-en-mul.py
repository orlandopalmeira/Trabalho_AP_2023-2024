import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Miguelcj1/model_opus-mt-en-mul-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

MAX_LENGTH = 128

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs, max_length=MAX_LENGTH, num_return_sequences=1)
    decoded_translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return decoded_translation[0]

def chatbot(message, target_language):
    if not target_language:
        return "Please choose a language to translate!"
    elif len(message) > MAX_LENGTH:
        return f"The input text can't be higher than {MAX_LENGTH}"
    else:
        translated_message = translate_text(">>" + language_options[target_language] + "<< " + message)
        return translated_message

language_options = {
    "French" : "fra",
    "Spanish" : "spa", 
    "German" : "deu", 
    "Portuguese" : "por"
}

demo_chatbot = gr.Interface(chatbot, ["textbox", gr.Dropdown(list(language_options.keys()), value="Portuguese", label="Target Language")], "text", title="Translation Chatbot (opus-mt-en-mul)", description="Enter text in English and choose a target language for translation.")
demo_chatbot.launch()