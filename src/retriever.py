# Handles recipe retrieval
"""
This module provides a function to search recipes by user query using the FAISS index.
"""
from sentence_transformers import SentenceTransformer
from faiss_store import RecipeFaissStore  # <-- Add this line at the top

MODEL_NAME = 'all-MiniLM-L6-v2'

def search_recipes_by_text(query, top_k=5):
    """
    Embeds the query, searches the FAISS index, and returns top-k recipes.
    """
    # Load model and FAISS store
    model = SentenceTransformer(MODEL_NAME)
    store = RecipeFaissStore()
    store.load_embeddings()      # Always load recipes/embeddings first
    store.load_faiss_index()    # Then load the index
    # Embed the query
    query_emb = model.encode(query)
    # Search
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
