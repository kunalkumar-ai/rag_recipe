# FAISS vector store implementation
"""
Install requirements (run in your environment):
    pip install faiss-cpu numpy

This module loads recipe embeddings, builds a FAISS index, and provides search functionality.
"""
import json
import numpy as np
import faiss

EMBEDDINGS_PATH = '/Users/kunalkumar/CascadeProjects/rag_recipe/processed_recipes/recipes_with_embeddings.jsonl'
INDEX_PATH = '/Users/kunalkumar/CascadeProjects/rag_recipe/processed_recipes/faiss.index'

class RecipeFaissStore:
    def __init__(self, embeddings_path=EMBEDDINGS_PATH, index_path=INDEX_PATH):
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.recipes = []
        self.embeddings = None
        self.index = None

    def load_embeddings(self):
        self.recipes = []
        vectors = []
        with open(self.embeddings_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                recipe = json.loads(line)
                if 'embedding' in recipe and recipe['embedding'] is not None:
                    self.recipes.append(recipe)
                    vectors.append(recipe['embedding'])
        self.embeddings = np.array(vectors).astype('float32')
        print(f"Loaded {len(self.recipes)} recipes with embeddings.")

    def build_faiss_index(self):
        if self.embeddings is None:
            self.load_embeddings()
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.embeddings.shape[0]} vectors.")
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}")

    def load_faiss_index(self):
        self.index = faiss.read_index(self.index_path)
        print(f"FAISS index loaded from {self.index_path}")

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            self.load_faiss_index()
        query_vec = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query_vec, top_k)
        results = [self.recipes[idx] for idx in I[0]]
        return results

if __name__ == "__main__":
    store = RecipeFaissStore()
    store.load_embeddings()
    store.build_faiss_index()
    # Example search (replace with actual query embedding):
    # query_emb = store.embeddings[0]
    # results = store.search(query_emb, top_k=3)
    # for r in results:
    #     print(r['title'])
