import token
from miditok import REMI, TokenizerConfig, REMIPlus
from pathlib import Path
from miditoolkit import MidiFile
import os
import json

from nbformat import convert


config = TokenizerConfig(num_velocities=64 ,use_chords=True, use_tempos=True, use_programs=True)
emopia_tokenizer = REMIPlus(config)

emopia_tokenizer._load_params(("tokenizer_emopia.json"))



print(emopia_tokenizer._ids_to_tokens([4, 452, 221, 295]))

# vocab = tokenizer._create_base_vocabulary()
# print(tokenizer._ids_to_tokens([0,1,2,3,4]))

# midi = MidiFile("clean_midi-chunked/.38 Special/Caught Up In You_0.mid")
# # tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
# # print(tokens)
# # converted_back_midi = tokenizer(tokens)

# Tokenize a whole dataset and save it at Json files
# midi_paths = list(Path("EMOPIA_1.0/midis").glob("**/*.mid"))  # data augmentation on 2 pitch octaves, 1 velocity and 1 duration values
# tokenizer.tokenize_midi_dataset(midi_paths, Path("EMOPIA_tokenized"))
# tokenizer.save_params(Path("tokenizer_emopia.json"))



# def find_corrupted_json_files(directory):
#     corrupted_files = []
#     for json_file in Path(directory).glob('**/*.json'):
#         try:
#             with open(json_file, 'r') as file:
#                 json.load(file)
#         except json.JSONDecodeError:
#             corrupted_files.append(str(json_file))
#     return corrupted_files

# # Replace 'your_directory_path' with the path to your JSON files
# directory_path = 'clean_midi_tokenized'
# corrupted_json_files = find_corrupted_json_files(directory_path)

# print("Corrupted JSON Files:")
# for file in corrupted_json_files:
#     print(file)
# Mapping from IDs to tokens
# import json

# # Load the original JSON file
# with open('tokenizer_js_decode.json', 'r') as json_file:
#     id_to_token = json.load(json_file)

# # Create the reverse mapping: token to ID
# token_to_id = {}
# for k, v in id_to_token.items():
#     if isinstance(v, list):
#         for item in v:
#             # Handling duplicate tokens if they occur
#             if item in token_to_id:
#                 print(f"Duplicate token '{item}' found for IDs {token_to_id[item]} and {k}")
#             token_to_id[item] = k
#     else:
#         token_to_id[v] = k

# # Save the reverse mapping to a new JSON file
# with open('token_to_id.json', 'w') as json_file:
#     json.dump(token_to_id, json_file, indent=4)

# print("JSON file 'token_to_id.json' has been created with token to ID mappings.")

    
# def convert_tokens_to_remi(json_file_path):
#     # Load the tokenized data from the JSON file
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)
#         tokens = data['ids']  # Assuming the tokens are stored under the key 'ids'

#     # Convert the tokens back to REMI format (human-readable token names)
#     remi_tokens = tokenizer._ids_to_tokens(tokens)

#     return remi_tokens

# print(tokenizer.special_tokens_ids)

# remi1 = convert_tokens_to_remi("clean_midi_tokenized/.38 Special/Caught Up In You_0.json")
# print(remi1)

# print(tokenizer._tokens_to_ids(encoding_midi))

# print(tokenizer._tokens_to_ids(["PAD_None"]))
# tokenizer.save_params(Path("tokenizer.json"))

# tokenizer._load_params(Path("tokenizer.json"))
# print(tokenizer)
# midi_paths = list(Path("clean_midi_tokenized").glob("**/*.mid"))



