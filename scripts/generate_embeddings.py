# Script for generating embeddings
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_PATH = '/Users/kunalkumar/CascadeProjects/rag_recipe/processed_recipes/recipes_final.jsonl'
OUTPUT_PATH = '/Users/kunalkumar/CascadeProjects/rag_recipe/processed_recipes/recipes_with_embeddings.jsonl'

# Load the MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_embedding_text(recipe):
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    directions = recipe.get("directions", [])
    return f"{title} | Ingredients: {', '.join(ingredients)} | Steps: {', '.join(directions)}"

def main():
    recipes = []
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        for line in fin:
            recipes.append(json.loads(line))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for recipe in tqdm(recipes, desc="Embedding recipes"):
            text = build_embedding_text(recipe)
            embedding = model.encode(text).tolist()
            recipe["embedding"] = embedding
            fout.write(json.dumps(recipe, ensure_ascii=False) + "\n")
    print(f"Saved recipes with embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
