from typing import Union

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from os.path import dirname, abspath
import os

app = FastAPI()

d = dirname(abspath(__file__))  # /home directory
deploy_path = os.path.join(d, "model")

def load_mod(mod_name):
    path = os.path.join(deploy_path, mod_name)
    mod = SentenceTransformer(path)
    return mod

@app.get("/",tags=["Version"])
def Version():
    return {"Version": "1.0"}


@app.post("/transformer")
def read_item(input_str: str):
    transformer_model = load_mod('all-mpnet-base-v2')
    return str(transformer_model.encode(input_str)[0])