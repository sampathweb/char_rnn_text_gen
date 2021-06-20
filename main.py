from fastapi import FastAPI

from string import punctuation

import numpy as np
import torch
from models import AliceModel, sample

app = FastAPI()

# Instantiate the model with hyperparameters
model_state = torch.load("model_alice_state.pt", map_location=torch.device('cpu'))
char2int = model_state["char2int"]
int2char = model_state["int2char"]
VOCAB_SIZE = len(char2int)
model = AliceModel(input_size=VOCAB_SIZE, output_size=VOCAB_SIZE, hidden_dim=model_state["hidden_dim"], n_layers=model_state["n_layers"])
model.load_state_dict(model_state["model"])


@app.get("/")
async def root(start:str="Alice woke up ", out_len:int=2000):
    data = sample(model, char2int, int2char, out_len=out_len, start=start)
    return {"message": data}
