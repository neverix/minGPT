from sentence_transformers import SentenceTransformer
from spherecluster import SphericalKMeans
from more_itertools import chunked
from itertools import islice
from tqdm.auto import tqdm
import numpy as np
# import diskcache
import torch
import gc


class Clusterer(object):
    def __init__(self, n_clusters=20, batch_size=64, threshold=0.5):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.kmeans = SphericalKMeans(n_clusters)
        self.bs = batch_size
        self.threshold = threshold
    
    def fit(self, dl, train_for=2000):
        self.model.to("cuda")
        with torch.no_grad():
            embeddings = []
            for i in chunked(tqdm(islice((x[0] for x in dl), train_for), total=min(len(dl), train_for)), self.bs):
                embeddings += list(self.model.encode(i, device="cuda"))
        self.kmeans.fit(embeddings)
        self.model.to("cpu")

    def predict(self, dl):
        bs = dl.batch_size
        queue_x, queue_y = [], []
        for s, (x, y) in dl:
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                emb = self.model.encode(s, device="cuda")
            gc.collect()
            torch.cuda.empty_cache()
            centers = self.kmeans.cluster_centers_
            sims = (centers[None, :] * emb[:, None]).sum(axis=-1).max(axis=-1)
            cond = sims > self.threshold
            queue_x += list(x[cond])
            queue_y += list(y[cond])
            if len(queue_x) >= bs:
                yield torch.stack(queue_x[:bs]), torch.stack(queue_y[:bs])
                queue_x, queue_y = queue_x[bs:], queue_y[bs:]
