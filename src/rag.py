# Data models for recipes
from retriever import search_recipes_by_text
from generator import generate_llm_answer

def main():
    user_query = input("Enter your question: ")
    # Retrieve top 5 relevant recipes (change as needed)
    recipes = search_recipes_by_text(user_query, top_k=5)
    print("\nTop recipes retrieved:")
    for i, r in enumerate(recipes, 1):
        print(f"{i}. {r['title']}")
    # Generate LLM answer
    answer = generate_llm_answer(user_query, recipes)
    print("\nLLM Answer:\n", answer)

if __name__ == "__main__":
    main()