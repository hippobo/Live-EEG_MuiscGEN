from flask import Flask, render_template, request, send_file, jsonify, make_response, send_from_directory
from werkzeug.utils import secure_filename
import torch
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
from threading import Thread
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from scipy import signal, stats
import mne
from tkinter import *
import subprocess, time


eeg_status = {"message": "", "status": "idle"}
eeg_model = keras.models.load_model("./models/EmotionNetV2.h5")

PROCESSING_INTERVAL = 2

fs = 256
inputLength = 10.5 # Length of input in seconds
shiftLength = 5 # Time between epochs
samples = int(shiftLength * fs) # How many samples to gather in every cycle

bufferSize = int(128 * inputLength) # Size of buffer in samples. Enough to hold one set of downsampled input.

buffers = np.zeros((4, bufferSize)) # buffers for each of the four channels
emotions = [[0,0,0]]

# Push new data onto buffer, removing any old data on the end
def updateBuffer(buffer, newData):
    assert len(newData.shape) == len(buffer.shape) and buffer.shape[0] >= newData.shape[0], "Buffer shape ({}) and new data shape ({}) are not compatible.".format(buffer.shape, newData.shape)
    size = newData.shape[0]
    buffer[:-size] = buffer[size:]
    buffer[-size:] = newData
    return buffer



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
    return render_template('demo.html')

@app.route('/live_eeg')
def live_eeg():
    with open('./static/eeg_data/eeg_data_live.json', 'w') as json_file:
                json.dump({}, json_file)
    return render_template('live_eeg.html')

@app.route('/eeg_data_demo')
def eeg_data_demo():
    return send_from_directory('static', 'eeg_data/eeg_data_demo.json')


@app.route('/eeg_data_live')
def eeg_data_live():
    
    return send_from_directory('static', 'eeg_data/eeg_data_live.json')




def eeg_inference():
    global eeg_status, emotions

    # Get the streamed data from the Muse. Blue Muse must be streaming.
    process = subprocess.Popen("muselsl stream --address 00:55:DA:B5:D5:CF", shell=True)
    print('Waiting 10 seconds for EEG stream to start...')
    time.sleep(10)

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=10, minimum=1)
   

    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=int(inputLength))
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    fs = int(info.nominal_srate())

   
    try:
        eeg_status = {"message": "Connecting to EEG...", "status": "connecting"}
        while True:
            eeg_status = {"message": "Processing EEG data...", "status": "processing"}
            # Continuously update EEG data and attempt to measure emotions
            
            data, timestamp = inlet.pull_chunk(timeout=5, max_samples=samples)
            eeg = np.array(data).swapaxes(0,1)

            """
            The DEAP dataset has 3 processing steps which we must also apply to out data:
            1. downsample to 128 Hz.
            2. Apply a bandpass frequency filter from 4-45 Hz
            3. Average data to the common reference
            """

            # Downsample
            processedEEG = signal.resample(eeg, int(eeg.shape[1] * (128 / fs)), axis=1)

            # Apply bandpass filter from 4-45Hz
            # processedEEG = mne.filter.filter_data(processedEEG, 128, 4, 45, filter_length=512, 
            #                             l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
            #                             phase='zero', fir_window='hamming', verbose=0)
            processedEEG = mne.filter.filter_data(processedEEG, 128, 4, 45, filter_length='auto', 
                                                l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                                                phase='zero', fir_window='hamming', verbose=0)


            
            # Zero mean
            processedEEG -= np.mean(processedEEG, axis=1, keepdims=True)
            
            

            # Update buffer
            for channel in range(buffers.shape[0]):
                buffers[channel] = updateBuffer(buffers[channel], processedEEG[channel])
            eeg_data_json = {
                "channel_data": [buffers[i].tolist() for i in range(4)]  # Data for each channel is a separate list
            }

            with open('./static/eeg_data/eeg_data_live.json', 'w') as json_file:
                json.dump(eeg_data_json, json_file)
            buffers_reshaped = np.expand_dims(buffers, axis=0)
            emotions = eeg_model.predict(buffers_reshaped).tolist()

            emotions = [[max(min(val, 9), 1) for val in emotion] for emotion in emotions]

            valence, arousal, dominance = emotions[0]
           
            print("Valence: {:.2f}".format(valence))
            print("Arousal: {:.2f}".format(arousal))
            print("Dominance: {:.2f}".format(dominance))
    except Exception as e:
        eeg_status = {"message": f"Error: {str(e)}", "status": "error"}
          
        
    finally:
        inlet.close_stream()
        with open('./static/eeg_data/eeg_data_live.json', 'w') as json_file:
            json.dump({}, json_file)
        eeg_status = {"message": "EEG stream closed", "status": "closed"}


@app.route('/eeg_emotions')
def get_eeg_emotions():
    global emotions
    return jsonify({
        "valence": emotions[0][0],
        "arousal": emotions[0][1],
        "dominance": emotions[0][2]
    })

@app.route('/connect_eeg')
def connect_eeg():
    try:
        # Start eeg_inference in a background thread
        thread = Thread(target=eeg_inference)
        thread.start()

        # Return a response immediately
        return jsonify({"status": "success", "message": "EEG connection initiated."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/eeg_status')
def get_eeg_status():
    return jsonify(eeg_status)

@app.route('/generate_midi_demo', methods=['POST'])
def generate_midi_demo():

    data = request.json
    
    sequence_length = data['sequence_length']
    context = data['context'] 
    quadrant_use = data['quadrant_use']
    condition_values = data['quadrant_counts']
    temperature_value = data['temperatureValue']
   
   

    if not quadrant_use:
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)
        model_noVA.load_state_dict(torch.load(f"models/model_{3}.pt", map_location=device))
        model_noVA.eval()
        output_directory = 'generated_midi'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        generated_seq = model_noVA.generate(context, max_new_tokens=sequence_length, temperature=temperature_value)

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
        generated_seq = model_emopia.generate(context, condition=torch.tensor(data=condition_values, dtype=torch.float, device=device), max_new_tokens=sequence_length, temperature=temperature_value)
 
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

@app.route('/generate_midi_live', methods=['POST'])
def generate_midi_live():
    data = request.json
    
    sequence_length = data['sequence_length']
    context = data['context']
    condition_values = data['quadrant_counts']
    temperature_value = data['temperatureValue']
    
    context_tensor = torch.tensor(context, dtype=torch.long, device=device)
    context_tensor = context_tensor.unsqueeze(0)

    model_emopia.load_state_dict(torch.load(f"models/model_emopia.pt", map_location=device))
    model_emopia.eval()

    # Generate new sequence tokens
    with torch.no_grad():
        new_seq = model_emopia.generate(context_tensor, condition=torch.tensor(data=condition_values, dtype=torch.float, device=device), max_new_tokens=sequence_length, temperature=temperature_value)
    
    # Convert only the new tokens to MIDI
    new_tokens = new_seq[:, len(context):].tolist()[0]  # Extract only the new tokens
    midi = emopia_tokenizer.tokens_to_midi(new_tokens)
    
    output_directory = 'generated_midi'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_name = secure_filename(f"generated_midi_seq.mid")
    output_file_path = os.path.join(output_directory, output_file_name)
    midi.dump(output_file_path)
      
    response_data = {
        'context': new_tokens,  # Include only the new sequence tokens
        'midi_file_url': f"/download_midi/{output_file_name}"  # URL to download the new MIDI file
    }

    return make_response(jsonify(response_data))




@app.route('/download_midi/<filename>')
def download_midi(filename):
    output_directory = 'generated_midi'
    return send_file(os.path.join(output_directory, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
  