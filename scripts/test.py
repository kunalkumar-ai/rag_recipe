import pandas as pd

df = pd.read_csv('/Users/kunalkumar/CascadeProjects/rag_recipe/data/raw_recipes/RecipeNLG_dataset.csv', nrows=1)
print(df.columns.tolist())