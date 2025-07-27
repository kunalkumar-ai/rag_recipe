# Handles recipe retrieval
"""
This module provides a function to search recipes by user query using the FAISS index.
"""
from sentence_transformers import SentenceTransformer
from faiss_store import RecipeFaissStore

MODEL_NAME = 'all-MiniLM-L6-v2'

# Load model and FAISS store ONCE at module load time
model = SentenceTransformer(MODEL_NAME)
store = RecipeFaissStore()
store.load_embeddings()      # Always load recipes/embeddings first
store.load_faiss_index()    # Then load the index

def search_recipes_by_text(query, top_k=5):
    """
    Embeds the query, searches the FAISS index, and returns top-k recipes.
    """
    # Only embed and search each time (fast)
    query_emb = model.encode(query)
    results = store.search(query_emb, top_k=top_k)
    return results

if __name__ == "__main__":
    query = input("Enter your recipe search query: ")
    recipes = search_recipes_by_text(query, top_k=5)
    print("Top matching recipes:")
    for i, recipe in enumerate(recipes, 1):
        print(f"{i}. {recipe['title']}")
        print(f"   Ingredients: {', '.join(recipe.get('ingredients', []))}")
        print()
