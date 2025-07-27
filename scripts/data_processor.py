# Script for processing raw data
import json
import ast
from pathlib import Path
from dietary_forbidden_lists import (
    FORBIDDEN_MEAT, FORBIDDEN_DAIRY, FORBIDDEN_EGGS, FORBIDDEN_HONEY
)

INPUT_PATH = "../processed_recipes/recipes_cleaned.jsonl"
OUTPUT_PATH = "../processed_recipes/recipes_final.jsonl"

def parse_stringified_list(s):
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []

def clean_ingredients(ingredients):
    # Lowercase, strip, remove empty
    return [i.strip().lower() for i in ingredients if i and isinstance(i, str) and i.strip()]

def clean_directions(directions):
    # Strip, remove empty
    return [step.strip() for step in directions if step and isinstance(step, str) and step.strip()]

def is_valid(recipe):
    # Drop if any of title, ingredients, directions is empty/null
    if not recipe.get("title"):
        return False
    if not recipe.get("ingredients") or len(recipe["ingredients"]) == 0:
        return False
    if not recipe.get("directions") or len(recipe["directions"]) == 0:
        return False
    return True

def contains_forbidden(ingredients, forbidden_set):
    for ingredient in ingredients:
        for forbidden in forbidden_set:
            if forbidden in ingredient:
                return True
    return False

def infer_dietary_tags(ingredients):
    tags = []
    # Vegan: no meat, dairy, eggs, honey
    if not (
        contains_forbidden(ingredients, FORBIDDEN_MEAT)
        or contains_forbidden(ingredients, FORBIDDEN_DAIRY)
        or contains_forbidden(ingredients, FORBIDDEN_EGGS)
        or contains_forbidden(ingredients, FORBIDDEN_HONEY)
    ):
        tags.append("vegan")
    # Vegetarian: no meat
    elif not contains_forbidden(ingredients, FORBIDDEN_MEAT):
        tags.append("vegetarian")
    return tags

def process_recipes(input_path, output_path):
    seen_titles = set()
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            recipe = json.loads(line)
            # Parse lists
            recipe["ingredients"] = parse_stringified_list(recipe.get("ingredients", "[]"))
            recipe["directions"] = parse_stringified_list(recipe.get("directions", "[]"))
            recipe["NER"] = parse_stringified_list(recipe.get("NER", "[]"))
            # Clean
            recipe["ingredients"] = clean_ingredients(recipe["ingredients"])
            recipe["directions"] = clean_directions(recipe["directions"])
            # Add num_ingredients and num_steps fields
            recipe["num_ingredients"] = len(recipe["ingredients"])
            recipe["num_steps"] = len(recipe["directions"])
            # Add dietary type tags
            recipe["type"] = infer_dietary_tags(recipe["ingredients"])
            # Remove duplicate titles
            title = recipe.get("title", "").strip().lower()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            # Drop if any required field is empty
            if not is_valid(recipe):
                continue
            fout.write(json.dumps(recipe, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_recipes(INPUT_PATH, OUTPUT_PATH)
    print(f"Processed recipes saved to {OUTPUT_PATH}")