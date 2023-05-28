from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

app = Flask(__name__)

# Define the paths to the saved model and tokenizer
model_path = 'C:/Users/Nada/OneDrive/Desktop/gl3/2eme semestre/ppp/hedhi nhessha bech tenjah/model'
tokenizer_path = 'C:/Users/Nada/OneDrive/Desktop/gl3/2eme semestre/ppp/hedhi nhessha bech tenjah/tokens'

# Load the pretrained model and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# API route for generating a poem
@app.route('/generate_poem', methods=['POST'])
def generate_poem():
    # Get the input text from the request body
    input_text = request.json['input_text']

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the poem
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    

    # Decode the generated output
    generated_poem = tokenizer.decode(output[0], skip_special_tokens=True)
    
    generated_poem=generated_poem[:generated_poem.rfind('.')]
    

    # Return the generated poem as a JSON response
    return jsonify({'poem': generated_poem})

if __name__ == '__main__':
    app.run()
