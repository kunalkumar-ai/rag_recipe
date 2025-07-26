import os
import json
import pandas as pd
from pathlib import Path
from typing import Iterator, Dict, Any, List

# --- Configuration ---
RAW_DATA_DIR = "../data/raw_recipes"
PROCESSED_DIR = "../processed_recipes"
INPUT_FILENAME = "RecipeNLG_dataset.csv"
SAMPLE_OUTPUT_FILENAME = "recipes_sample.jsonl"
FULL_OUTPUT_FILENAME = "recipes_cleaned.jsonl"
SAMPLE_SIZE = 100
CHUNKSIZE = 1000

# --- Helper Functions ---

def clean_output_dir(directory: str) -> None:
    """Delete all files in the output directory, but do not remove the directory itself."""
    dir_path = Path(directory)
    if dir_path.exists() and dir_path.is_dir():
        for file in dir_path.iterdir():
            if file.is_file():
                file.unlink()
    else:
        dir_path.mkdir(parents=True, exist_ok=True)

def stream_cleaned_recipes(file_path: str) -> Iterator[Dict[str, Any]]:
    """Load, clean, and yield recipes one by one from the CSV, dropping the Unnamed: 0 column if present."""
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNKSIZE, low_memory=False):
            # Remove Unnamed: 0 column if it exists
            if 'Unnamed: 0' in chunk.columns:
                chunk = chunk.drop(columns=['Unnamed: 0'])
            # Basic cleaning: drop rows with missing essential data
            chunk = chunk.dropna(subset=['title', 'ingredients', 'directions'])
            # Convert to dictionary records and yield each one
            for record in chunk.to_dict(orient='records'):
                yield record
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return

def save_recipes_to_jsonl(recipes: Iterator[Dict[str, Any]], output_path: str) -> int:
    """Save an iterator of recipes to a JSON Lines file."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for recipe in recipes:
            f.write(json.dumps(recipe, ensure_ascii=False) + '\n')
            count += 1
    return count

# --- Main Processing Functions ---

def create_sample_file(input_path: str, sample_path: str) -> List[Dict[str, Any]]:
    """Create and save a sample of recipes."""
    print(f"Creating sample file at {sample_path}...")
    recipe_stream = stream_cleaned_recipes(input_path)
    sample_recipes = []
    try:
        for _ in range(SAMPLE_SIZE):
            sample_recipes.append(next(recipe_stream))
    except StopIteration:
        pass # Handle cases where the dataset has fewer than SAMPLE_SIZE recipes
    
    with open(sample_path, 'w', encoding='utf-8') as f:
        for recipe in sample_recipes:
            f.write(json.dumps(recipe, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(sample_recipes)} recipes to sample file.")
    return sample_recipes

def process_full_dataset(input_path: str, full_output_path: str):
    """Process the entire dataset and save it to a new file."""
    print(f"Processing full dataset and saving to {full_output_path}...")
    recipe_stream = stream_cleaned_recipes(input_path)
    total_recipes = save_recipes_to_jsonl(recipe_stream, full_output_path)
    print(f"Processed and saved {total_recipes} recipes.")

def main():
    """Main function to orchestrate the data processing."""
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    
    # Define paths relative to the script's location
    input_file = script_dir / RAW_DATA_DIR / INPUT_FILENAME
    processed_dir_path = script_dir / PROCESSED_DIR
    sample_output = processed_dir_path / SAMPLE_OUTPUT_FILENAME
    full_output = processed_dir_path / FULL_OUTPUT_FILENAME
    
    # Clean (empty) the output directory, but do not recreate if it exists
    clean_output_dir(processed_dir_path)
    
    # --- Step 1: Create and inspect a sample ---
    sample_recipes = create_sample_file(input_file, sample_output)
    
    if not sample_recipes:
        print("No recipes could be processed. Please check the input file and its format.")
        return

    # Print summary from the sample
    print("\n--- Data Inspection (from sample) ---")
    print("Columns found:")
    print(list(sample_recipes[0].keys()))
    print("\nSample recipe:")
    print(json.dumps(sample_recipes[0], indent=2, ensure_ascii=False))
    print("-------------------------------------\n")
    
    # --- Step 2: Process the full dataset ---
    process_full_dataset(input_file, full_output)
    
    print("\nData collection and preprocessing complete.")

if __name__ == "__main__":
    main()
