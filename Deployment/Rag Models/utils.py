import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class PassageIndex:
    def __init__(self, parquet_file: str):
        self.data = pd.read_parquet(parquet_file)
        self.passages = (self.data['question']+" "+ self.data['answer']).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.embeddings = None
        self.build_index()

    def embed_passages(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.encoder(**tokens)
        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

    def build_index(self):
        self.embeddings = self.embed_passages(self.passages)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, top_k=3):
        query_vec = self.embed_passages([query])
        distances, indices = self.index.search(query_vec, top_k)
        return [self.passages[i] for i in indices[0]]