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
import math
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
from miditoolkit import MidiFile
import json
from torch.utils.data import Dataset
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
import onnxruntime as ort
import numpy as np

config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
clean_tokenizer = REMI(config)

clean_tokenizer._load_params(("tokenizer.json"))

clean_vocab = clean_tokenizer._create_base_vocabulary()
vocab_size_clean = len(clean_vocab) + 4 # +4 for special tokens
print("vocab size clean_midi : ", vocab_size_clean)

block_size = 256
batch_size = 64
n_embd = 128
d_ff = 4 * n_embd
n_head = 6
dropout = 0.2
device = torch.device("cuda")

# max_iters = 100
MAX_LEN= 500
num_epochs = 5
learning_rate = 2e-5
n_layer = 6


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
class CleanMidiDataset_DecoderOnly(Dataset):
    def __init__(self, path_to_dataset, dataset_fraction, unit_size=block_size):
        self.unit_size = unit_size
        self.path_to_dataset = path_to_dataset
        self.dataset_fraction = dataset_fraction

        encoded_midi_paths = list(Path(path_to_dataset).glob("**/*.json"))
        partial_data = round(len(encoded_midi_paths) * dataset_fraction)
        self.data_paths = encoded_midi_paths[:partial_data]

        self.all_samples = []
        for path in tqdm(self.data_paths, desc='Loading data'):
            with open(path, 'r') as f:
                data = json.load(f)
                song = data['ids']
                # Split the song into segments of `unit_size` for training
                for i in range(0, len(song) - unit_size, unit_size):
                    segment = song[i:i + unit_size]
                    self.all_samples.append(segment)
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        x = sample[:-1]
        y = sample[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



def temperature_scaled_softmax(logits, temperature):
    scaled_logits = logits / temperature
    softmax = torch.nn.Softmax(dim=-1)
    return softmax(scaled_logits)


class GeneratorModelDecoder(nn.Module):
    def __init__(self, n_embd, n_head, vocab_size=348, n_layer=n_layer):
        super(GeneratorModelDecoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)



    def _init_weights(self, module):
        # Initialize weights as per the paper
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
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

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = temperature_scaled_softmax(logits,temperature=0.7) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        

            # idx_next = nucleus_sampling(probs, p=0.7)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def cross_entropy_loss(self, pred_logits, target):
        # Reshape pred_logits to [batch_size * sequence_length, num_classes]
        pred_logits = pred_logits.view(-1, self.vocab_size)  

        # Reshape target to [batch_size * sequence_length]
        target = target.view(-1)  # Assuming target is [batch_size, sequence_length]

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(pred_logits, target, ignore_index=0, label_smoothing=0.1)
        
        return ce_loss
    

def nucleus_sampling(probs, p=0.9):
    """
    Apply Nucleus (Top-p) sampling to the logits.

    Args:
    - logits (torch.Tensor): Logits output by the model for the next token prediction.
    - p (float): Cumulative probability threshold for nucleus sampling, typically between 0.9 and 1.0.

    Returns:
    - selected_token (int): The index of the next token sampled.
    """


    # Sort the probabilities to identify the top-p tokens
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum of sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set the probabilities of removed tokens to zero
    sorted_probs[sorted_indices_to_remove] = 0

    # Resample from the adjusted probability distribution
    selected_token = torch.multinomial(sorted_probs, 1)

    return selected_token

model = GeneratorModelDecoder(n_embd, n_head).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# clean_midi_train = CleanMidiDataset_DecoderOnly('clean_midi_full_tokenized', 0.6)
# clean_midi_train_dataloader = DataLoader(clean_midi_train, batch_size=batch_size, shuffle=True)
# criterion = model.cross_entropy_loss

# steps = 0

# for epoch in range(num_epochs):
#     tepoch = tqdm(clean_midi_train_dataloader, desc=f'Training Epoch {epoch+1}', leave=False)
#     model.train()
#     for xb, yb in tepoch:
#         xb, yb = xb.to(device), yb.to(device)
#         steps +=1
#         logits, loss = model(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         tepoch.set_postfix(loss=f'{loss.item():.4f}')
#         if steps % 10000 == 0:
#             torch.save(model.state_dict(), f"models/model_{3}.pt")





    # torch.save(model.state_dict(), f"models/model_{3}.pt")

model.load_state_dict(torch.load( f"models/model_{3}.pt", map_location=device ))
model.eval()



# dummy_input = torch.randint(0, vocab_size_clean, (1,block_size)).to(device)

# torch.onnx.export(model,
#                   dummy_input,
#                   "models/musicgen.onnx",
#                   input_names=['input'],   # Replace with your actual input name
#                   output_names=['output'], # Replace with your actual output name
#                   dynamic_axes={'input': {1: 'sequence'},
#                                 'output': { 1: 'sequence'}})




def temperature_scaled_softmax_np(logits, temperature=0.7):
    probs = np.exp(logits / temperature)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs

def generate_sequence_with_onnx(session, initial_context, max_new_tokens, block_size):
    generated_sequence = [initial_context]
    current_context = np.full((1, block_size), 173, dtype=np.int64)
    current_context[0, 0] = initial_context

    for _ in range(max_new_tokens):
        # Get logits from the model
        logits = session.run(None, {'input': current_context})[0]
        # Focus on the last time step and apply softmax
        last_logits = logits[:, -1, :]
        probs = temperature_scaled_softmax_np(last_logits)

        # Sample from the multinomial distribution
        idx_next = np.random.choice(np.arange(probs.shape[-1]), p=probs[0])

        # Append sampled index to the running sequence
        generated_sequence.append(idx_next)

        # Update current context
        current_context[0, 1:] = current_context[0, :-1]
        current_context[0, 0] = idx_next
        print(current_context)

    return generated_sequence

# Load the ONNX model
# onnx_session = ort.InferenceSession("models/musicgen2.onnx")

# Generation
num_generations = 1
# for i in range(num_generations):
#     initial_context = 173  # Starting token
#     generated_seq = generate_sequence_with_onnx(onnx_session, initial_context, MAX_LEN, block_size)

#     # Decode the sequence
#     generated_text = clean_tokenizer._ids_to_tokens(generated_seq)
for i in range(num_generations):
    output_directory = 'generated_midi'

    context = torch.full((1, 1), 173, dtype=torch.long, device=device)
    generated_seq = model.generate(context, max_new_tokens=250)
    
    generated_text = clean_tokenizer._ids_to_tokens(generated_seq.tolist()[0])
    
    midi = clean_tokenizer.tokens_to_midi(generated_seq.tolist()[0])
    output_file_path = os.path.join(output_directory, f"output_{2}.mid")
    midi.dump(output_file_path)


# midi_seq = [
#     "Position_0", "Program_-1", "Pitch_42", "Velocity_79", "Duration_0.1.8", 
#     "Position_4", "Program_5", "Pitch_60", "Velocity_55", "Duration_0.3.8", 
#     "Position_5", "Program_5", "Pitch_65", "Velocity_71", "Duration_0.2.8", 
#     "Position_7", "Program_5", "Pitch_67", "Velocity_63", "Duration_0.4.8", 
#     "Program_5", "Pitch_76", "Velocity_63", "Duration_1.7.8", 
#     "Program_5", "Pitch_84", "Velocity_55", "Duration_1.6.8", 
#     "Program_25", "Pitch_72", "Velocity_39", "Duration_0.3.8", 
#     "Program_25", "Pitch_77", "Velocity_39", "Duration_0.3.8", 
#     "Program_25", "Pitch_81", "Velocity_39", "Duration_0.3.8", 
#     "Position_7", "Program_25", "Pitch_67", "Velocity_55", "Duration_0.4.8", 
#     "Program_26", "Pitch_64", "Velocity_71", "Duration_1.5.8", 
#     "Program_26", "Pitch_72", "Velocity_47", "Duration_2.5.8", 
#     "Position_8", "Program_5", "Pitch_62", "Velocity_63", "Duration_0.4.8", 
#     "Program_25", "Pitch_60", "Velocity_55", "Duration_0.3.8", 
#     "Program_26", "Pitch_60", "Velocity_55", "Duration_1.2.8", 
#     "Program_35", "Pitch_36", "Velocity_79", "Duration_0.3.8", 
#     "Program_26", "Pitch_72", "Velocity_55", "Duration_2.2.8", 
#     "Program_-1", "Pitch_40", "Velocity_63", "Duration_0.1.8", 
#     "Program_26", "Pitch_64", "Velocity_55", "Duration_3.1.8", 
#     "Position_15", "Program_5", "Pitch_86", "Velocity_63", "Duration_1.2.8", 
#     "Position_16", "Program_5", "Pitch_72", "Velocity_55", "Duration_1.3.8", 
#     "Program_35", "Pitch_36", "Velocity_71", "Duration_1.6.8", 
#     "Program_5", "Pitch_60", "Velocity_63", "Duration_1.2.8", 
#     "Program_26", "Pitch_60", "Velocity_55", "Duration_1.0.8", 
#     "Program_25", "Pitch_60", "Velocity_47", "Duration_0.7.8", 
#     "Program_26", "Pitch_60", "Velocity_55", "Duration_1.0.8", 
#     "Program_25", "Pitch_72", "Velocity_47", "Duration_0.3.8", 
#     "Program_26", "Pitch_64", "Velocity_47", "Duration_1.0.8", 
#     "Program_26", "Pitch_72", "Velocity_63", "Duration_7.3.4", 
#     "Program_-1", "Pitch_35", "Velocity_79", "Duration_0.1.8", 
#     "Position_17", "Program_25", "Pitch_84", "Velocity_63", "Duration_0.3.8", 
#     "Position_19", "Program_5", "Pitch_60", "Velocity_31", "Duration_0.4.8", 
#     "Position_20", "Program_25", "Pitch_76", "Velocity_63", "Duration_0.3.8", 
#     "Position_21", "Program_5", "Pitch_55", "Velocity_63", "Duration_0.4.8", 
#     "Program_25", "Pitch_67", "Velocity_55", "Duration_0.3.8", 
#     "Program_26", "Pitch_64", "Velocity_63", "Duration_0.3.8", 
#     "Program_25", "Pitch_76", "Velocity_55", "Duration_0.2.8", 
#     "Position_24", "Program_5", "Pitch_76", "Velocity_71", "Duration_0.3.8", 
#     "Program_5", "Pitch_84", "Velocity_63", "Duration_0.3.8", 
#     "Program_25", "Pitch_72", "Velocity_55", "Duration_0.3.8", 
#     "Program_25", "Pitch_76", "Velocity_55", "Duration_0.3.8", 
#     "Program_26", "Pitch_72", "Velocity_63", "Duration_0.3.8", 
#     "Program_-1", "Pitch_40", "Velocity_63", "Duration_0.1.8", 
#     "Position_29", "Program_5", "Pitch_84", "Velocity_71", "Duration_0.3.8", 
#     "Program_25", "Pitch_75", "Velocity_47", "Duration_0.3.8", 
#     "Program_26", "Pitch_72", "Velocity_63", "Duration_0.3.8", 
#     "Program_-1", "Pitch_40", "Velocity_63", "Duration_0.1.8", 
#     "Position_31", "Program_5", "Pitch_67", "Velocity_71", "Duration_0.2.8", 
#     "Program_26", "Pitch_67", "Velocity_63", "Duration_0.3.8", 
#     "Bar_None", "Position_0", "Program_5", "Pitch_57", "Velocity_47", "Duration_0.4.8", 
#     "Program_5", "Pitch_81", "Velocity_47", "Duration_0.3.8", 
#     "Program_5", "Pitch_93", "Velocity_63", "Duration_0.3.8", 
#     "Program_35", "Pitch_41", "Velocity_71", "Duration_1.2.8", 
#     "Program_50", "Pitch_69", "Velocity_47", "Duration_2.0.8", 
#     "Program_50", "Pitch_72", "Velocity_47", "Duration_2.0.8", 
#     "Program_50", "Pitch_77", "Velocity_47", "Duration_2.0.8", 
#     "Program_-1", "Pitch_35", "Velocity_79", "Duration_0.1.8", 
#     "Position_5", "Program_5", "Pitch_65", "Velocity_55", "Duration_0.3.8", 
#     "Program_25", "Pitch_77", "Velocity_55", "Duration_0.3.8", 
#     "Program_26", "Pitch_69"
# ]

# midi_seq = [clean_tokenizer._tokens_to_ids([tok])[0] for tok in midi_seq]

# midi = clean_tokenizer.tokens_to_midi(midi_seq)
# output_file_path = os.path.join("generated_midi", f"generated_midi_python.mid")
# midi.dump(output_file_path)
