from flask import Flask, render_template, request, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
import torch
from generatordecoder import GeneratorModelDecoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
from pathlib import Path
import os
import json
from torch.utils.data import Dataset


if torch.cuda.is_available():
    print("Using GPU", torch.cuda.is_available())
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

app = Flask(__name__) 
 



config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
clean_tokenizer = REMI(config)

clean_tokenizer._load_params(("tokenizer.json"))
 
clean_vocab = clean_tokenizer._create_base_vocabulary()
vocab_size_clean = len(clean_vocab) + 4 # +4 for special tokens
# print("vocab size clean_midi : ", vocab_size_clean)

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
     


model = GeneratorModelDecoder(n_embd, n_head).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_midi', methods=['POST'])
def generate_midi():
    data = request.json
    sequence_length = data['sequence_length']
    context = data['context']
    # context = clean_tokenizer._tokens_to_ids(context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)

    model.load_state_dict(torch.load(f"models/model_{3}.pt", map_location=device))
    model.eval()
 
    output_directory = 'generated_midi'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    generated_seq = model.generate(context, max_new_tokens=sequence_length)

    midi = clean_tokenizer.tokens_to_midi(generated_seq.tolist()[0])
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
  