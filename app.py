from flask import Flask, render_template, request, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
import torch
from generatordecoder import GeneratorModelDecoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from miditok import REMI, REMIPlus, TokenizerConfig
from pathlib import Path
import os
import json
from torch.utils.data import Dataset

from generatordecoder import GeneratorModelDecoder
from generatordecoder_EMOPIA import GeneratorModelDecoder_EMOPIA

if torch.cuda.is_available():
    print("Using GPU", torch.cuda.is_available())
    device = 'cuda'
else:
    device = 'cpu'

app = Flask(__name__) 
 


#  Base Decoder no VA
config_clean = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
clean_tokenizer = REMI(config_clean)

clean_tokenizer._load_params(("tokenizer.json"))
 

block_size = 256
n_embd = 128
n_head = 6

model_noVA = GeneratorModelDecoder(n_embd, n_head).to(device)


# EMOPIA Config

config_emopia = TokenizerConfig(num_velocities=64 ,use_chords=True, use_tempos=True, use_programs=True)
emopia_tokenizer = REMIPlus(config_emopia)

emopia_tokenizer._load_params(("tokenizer_emopia.json"))


n_embd_emopia = 576
n_head_emopia = 6


model_emopia = GeneratorModelDecoder_EMOPIA(n_embd_emopia, n_head_emopia).to(device)


@app.route('/')
def index():
    return render_template('index.html')
 
 
@app.route('/generate_midi', methods=['POST'])
def generate_midi():
    data = request.json
    sequence_length = data['sequence_length']
    context = data['context']
    quadrant_use = data['quadrant_use']
    condition_values = data['quadrant_counts']


    if not quadrant_use:
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)

        model_noVA.load_state_dict(torch.load(f"models/model_{3}.pt", map_location=device))
        model_noVA.eval()
    
        output_directory = 'generated_midi'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        generated_seq = model_noVA.generate(context, max_new_tokens=sequence_length)

        midi = clean_tokenizer.tokens_to_midi(generated_seq.tolist()[0])
        output_file_name = secure_filename(f"generated_midi_seq.mid")
        output_file_path = os.path.join(output_directory, output_file_name)
        midi.dump(output_file_path)
    else:  
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)

        model_emopia.load_state_dict(torch.load(f"models/model_emopia.pt", map_location=device))
        model_emopia.eval()
    
        output_directory = 'generated_midi'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        print(condition_values)
        generated_seq = model_emopia.generate(context, condition=torch.tensor(data=condition_values, dtype=torch.float, device=device), max_new_tokens=sequence_length)
 
        midi = emopia_tokenizer.tokens_to_midi(generated_seq.tolist()[0])
        output_file_name = secure_filename(f"generated_midi_seq.mid")
        output_file_path = os.path.join(output_directory, output_file_name)
        midi.dump(output_file_path)

    response_data = {
        'context': generated_seq.tolist()[0],  # Include the generated sequence tokens
        'midi_file_url': f"/download_midi/{output_file_name}"  # URL to download the MIDI file
    } 

    response = make_response(jsonify(response_data))
    return response

@app.route('/download_midi/<filename>')
def download_midi(filename):
    output_directory = 'generated_midi'
    return send_file(os.path.join(output_directory, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
  