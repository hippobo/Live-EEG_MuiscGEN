from distutils.command import clean
from functools import partial
from re import T
import datasets
from networkx import triangles
import transformers
import tokenizers
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
import torch
import torch.nn as nn
import math as m
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
from miditok import REMI, TokenizerConfig, REMIPlus
from pathlib import Path
import os
from miditoolkit import MidiFile
import json
from torch.utils.data import Dataset
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
import onnxruntime as ort
import numpy as np
import pandas as pd

config = TokenizerConfig(num_velocities=64 ,use_chords=True, use_tempos=True, use_programs=True)
emopia_tokenizer = REMIPlus(config)

emopia_tokenizer._load_params(("tokenizer_emopia.json"))

clean_vocab = emopia_tokenizer._create_base_vocabulary()
vocab_size_emopia = len(clean_vocab) + 4 # +4 for special tokens
print("vocab size emopia : ", vocab_size_emopia)

block_size = 256
batch_size = 64
n_embd = 576
d_ff = 4 * n_embd
n_head = 6
dropout = 0.2
device = torch.device("cuda")

# max_iters = 100
MAX_LEN= 500
num_epochs = 50
learning_rate = 2e-4
n_layer = 8


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EMOPIA(Dataset):
    def __init__(self, path_to_dataset, path_to_csv, dataset_fraction, unit_size=block_size):
        self.unit_size = unit_size
        self.path_to_dataset = path_to_dataset
        self.dataset_fraction = dataset_fraction

        # Load CSV and create a dictionary mapping songID to features
        df = pd.read_csv(path_to_csv)
        self.song_features = df.set_index('songID').to_dict('index')

        # Set of songIDs in the training set
        train_song_ids = set(df['songID'])

        # Load and process JSON files
        all_encoded_midi_paths = list(Path(path_to_dataset).glob("**/*.json"))
        # Filter paths to include only those containing a song ID from the training set
        encoded_midi_paths = [path for path in all_encoded_midi_paths if any(song_id in path.stem for song_id in train_song_ids)]

        partial_data = round(len(encoded_midi_paths) * dataset_fraction)
        self.data_paths = encoded_midi_paths[:partial_data]

        self.all_samples = []
        for path in tqdm(self.data_paths, desc='Loading data'):
            with open(path, 'r') as f:
                data = json.load(f)
                # Extract song_id from filename
                song_id = self.extract_song_id(path.stem, train_song_ids)
                song = data['ids']
                if song_id and song_id in self.song_features:  # Check if the extracted songID is valid
                    features = self.song_features[song_id]
                    for i in range(0, len(song) - unit_size, unit_size):
                        segment = song[i:i + unit_size]
                        self.all_samples.append((segment, features))

    def extract_song_id(self, filename, song_ids):
        
        for song_id in song_ids:
            if song_id in filename:
                return song_id
        return None

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample, features = self.all_samples[idx]
        x = sample[:-1]
        y = sample[1:]
        # features_vector = torch.tensor([features['num_Q1'], features['num_Q2'], features['num_Q3'], features['num_Q4'], features['DominantQ']], dtype=torch.float)
        features_vector = torch.tensor([features['num_Q1'], features['num_Q2'], features['num_Q3'], features['num_Q4']], dtype=torch.float)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), features_vector

def temperature_scaled_softmax(logits, temperature):
    scaled_logits = logits / temperature
    softmax = torch.nn.Softmax(dim=-1)
    return softmax(scaled_logits)



def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        embed_sinusoid_list = sinusoid(max_seq, embedding_dim)

        self.positional_embedding = torch.from_numpy(embed_sinusoid_list).to(
            self.device, dtype=self.dtype)

    def forward(self, x):
        if x.device != self.device or x.dtype != self.dtype:
            self.positional_embedding = self.positional_embedding.to(x.device, dtype=x.dtype)
        x += self.positional_embedding[:, :x.size(1), :]
        return x

class GeneratorModelDecoder_EMOPIA(nn.Module):
    def __init__(self, n_embd, n_head, vocab_size=vocab_size_emopia, n_layer=n_layer,d_condition=192):
        super(GeneratorModelDecoder_EMOPIA, self).__init__()
        self.d_condition = d_condition
        
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim=n_embd-self.d_condition)
        self.pos_encoding = DynamicPositionEmbedding(n_embd, max_seq=MAX_LEN)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.fc_condition = nn.Linear(4, d_condition)

        self.apply(self._init_weights)



    def _init_weights(self, module):
        # Initialize weights as per the paper
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, condition=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb  # (B,T,C)
        x *= m.sqrt(n_embd-self.d_condition)

        if condition is not None:
            condition = self.fc_condition(condition)  # (B, d_condition)
            condition = condition.view(B, 1, self.d_condition)  # Reshape condition to (B, 1, d_condition)
            condition = condition.expand(-1, T, -1)  # Expand to match x's sequence length (B, T, d_condition)
            x = torch.cat([x, condition], dim=-1)  # Concatenate condition

        x = self.pos_encoding(x)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, condition, max_new_tokens, temperature=0.7):
        """
        Generate a sequence of tokens given an initial context and condition.
        :param idx: (B, T) tensor of initial token indices
        :param condition: (B, condition_dim) tensor of condition values
        :param max_new_tokens: the maximum number of tokens to generate
        :param temperature: the temperature for scaling the softmax distribution
        :return: generated sequence of token indices
        """
        B, _ = idx.shape
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens, if it's longer
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx

            # Get the predictions
            logits, _ = self(idx_cond, condition=condition)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # focus only on the last time step (B, vocab_size)

            # Apply temperature-scaled softmax to get probabilities
            probs = temperature_scaled_softmax(logits, temperature)  # (B, vocab_size)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            
            # condition = update_condition(condition)

        return idx

    def cross_entropy_loss(self, pred_logits, target):
        # Reshape pred_logits to [batch_size * sequence_length, num_classes]
        pred_logits = pred_logits.view(-1, self.vocab_size)  

        # Reshape target to [batch_size * sequence_length]
        target = target.view(-1)  # Assuming target is [batch_size, sequence_length]

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(pred_logits, target, ignore_index=0, label_smoothing=0.1)
        
        return ce_loss
    


# model = GeneratorModelDecoder_EMOPIA(n_embd, n_head).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# emopia_midi_train = EMOPIA('EMOPIA_tokenized', 'EMOPIA_1.0/split/train_SL.csv', 1)
# emopia_midi_test = EMOPIA('EMOPIA_tokenized', 'EMOPIA_1.0/split/test_SL.csv', 1)
# emopia_midi_val = EMOPIA('EMOPIA_tokenized', 'EMOPIA_1.0/split/val_SL.csv', 1)


# emopia_train_dataloader = DataLoader(emopia_midi_train, batch_size=batch_size, shuffle=True)
# emopia_test_dataloader = DataLoader(emopia_midi_test, batch_size=batch_size, shuffle=False)
# emopia_val_dataloader = DataLoader(emopia_midi_val, batch_size=batch_size, shuffle=False)





# clean_midi_train = CleanMidiDataset_DecoderOnly('clean_midi_full_tokenized', 0.6)
# clean_midi_train_dataloader = DataLoader(clean_midi_train, batch_size=batch_size, shuffle=True)

# steps = 0

# for epoch in range(num_epochs):
#     tepoch = tqdm(emopia_train_dataloader, desc=f'Training Epoch {epoch+1}', leave=False)
#     model.train()
#     for xb, yb, condition in tepoch:
#         xb, yb , condition= xb.to(device), yb.to(device), condition.to(device)
#         steps +=1
#         logits, loss = model(idx=xb, targets=yb, condition=condition)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         tepoch.set_postfix(loss=f'{loss.item():.4f}')
#         if steps % 60 == 0:
#             torch.save(model.state_dict(), f"models/model_emopia{steps}.pt")





#     torch.save(model.state_dict(), f"models/model_emopia.pt")

# model.load_state_dict(torch.load( f"models/model_emopia.pt", map_location=device ))
# model.eval()





# def temperature_scaled_softmax_np(logits, temperature=0.7):
#     assert temperature > 0, "Temperature must be positive"
#     probs = torch.exp(logits / temperature)
#     probs /= torch.sum(probs, dim=-1, keepdim=True)
#     return probs

# # Generation
# num_generations = 1
# # for i in range(num_generations):
# #     initial_context = 173  # Starting token
# #     generated_seq = generate_sequence_with_onnx(onnx_session, initial_context, MAX_LEN, block_size)

# #     # Decode the sequence
# #     generated_text = clean_tokenizer._ids_to_tokens(generated_seq)
# for i in range(num_generations):
#     output_directory = 'generated_midi'

#     context = torch.full((1, 1), 173, dtype=torch.long, device=device)
#     generated_seq = model.generate(idx=context, condition=torch.tensor(data=[2.,1.,0.,0.], dtype=torch.float, device=device), max_new_tokens=300)
    
#     generated_text = emopia_tokenizer._ids_to_tokens(generated_seq.tolist()[0])
    
#     midi = emopia_tokenizer.tokens_to_midi(generated_seq.tolist()[0])
#     output_file_path = os.path.join(output_directory, f"output_{5}.mid")
#     midi.dump(output_file_path)

