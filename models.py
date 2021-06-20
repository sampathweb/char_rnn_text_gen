import numpy as np

import torch
from torch import nn

class AliceModel(nn.Module):
    """Creates a RNN Model with n_layers and hidden_dim"""
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, hidden=None):
        
        batch_size = x.size(0)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden.detach()


def one_hot_encode(sequence, vocab_size):
    """Inputs need to be converted to one-hot encoded vectors to feed into RNN"""
    # Creating a multi-dimensional array of zeros with the desired output shape
    # (Sequence Length, One-Hot Encoding Size)
    seq_length = len(sequence)
    output = np.zeros((seq_length, vocab_size), dtype=np.float32)

    for seq in range(seq_length):
      output[seq, sequence[seq]] = 1
    
    return output

def predict(model, hidden, character, char2int, int2char, device):
    # One-hot encoding our input to fit into the model
    # print(character)
    character = np.array([char2int[c] for c in character])
    # print(character)
    character = one_hot_encode(character, vocab_size=len(char2int))
    # print(character.shape)
    character = torch.from_numpy(character).unsqueeze(0).to(device)
    with torch.no_grad():
      out, hidden = model(character, hidden)
    # print(hidden.size())
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden


def sample(model, char2int, int2char, out_len, start='hey', device=None):
    model.eval() # eval mode
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"    
    model.to(device)
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    hidden = None
    char, hidden = predict(model, hidden, chars, char2int, int2char, device)
    for ii in range(size):
        # Use last char and hidden state to generate next char
        char, hidden = predict(model, hidden, chars[-1], char2int, int2char, device)
        chars.append(char)  # Add the predicted char to output
    return ''.join(chars)
