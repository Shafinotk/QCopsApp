# utils/embeddings.py
import os
import numpy as np

# We use streamlit caches in app.py to hold the model, but provide plain functions here.
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a loaded SentenceTransformer model. Caller may wrap/cached this.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model

def batch_encode(texts, model=None, batch_size=64):
    """
    Encodes a list of texts into embeddings (np.float32).
    If model is None, loads a default model (slightly slower).
    """
    if model is None:
        model = load_embedding_model()
    # Filter and encode in batches
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs)
    if all_embs:
        arr = np.vstack(all_embs).astype("float32")
    else:
        arr = np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return arr

def build_faiss_index(embeddings):
    """
    Build a simple FAISS index using inner-product similarity.
    Caller should normalize the embeddings beforehand if desired.
    """
    try:
        import faiss
    except Exception as e:
        raise ImportError("faiss is required for building an index. Install faiss-cpu.") from e
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Empty embeddings provided")
    d = embeddings.shape[1]
    # normalize for inner product (cosine)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    import faiss
    faiss.write_index(index, path)

def load_faiss_index(path):
    import faiss
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    index = faiss.read_index(path)
    return index

def query_faiss_index(index, query_embs, top_k=5):
    """
    query_embs: np.array shape (nq, d)
    Returns (distances, indices)
    """
    import faiss
    if query_embs is None or len(query_embs) == 0:
        return [], []
    faiss.normalize_L2(query_embs)
    distances, indices = index.search(query_embs.astype("float32"), top_k)
    return distances, indices
