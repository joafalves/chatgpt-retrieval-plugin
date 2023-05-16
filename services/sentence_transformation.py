from typing import List
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts: List[str]) -> List[List[float]]:
    print(f"Local embedding for {len(texts)}")
    return model.encode(texts).tolist()
