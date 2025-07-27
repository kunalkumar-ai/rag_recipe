# Recipe generation logic
import os
from dotenv import load_dotenv
import google.generativeai as genai

def format_recipes_for_context(recipes, max_recipes=5):
    """Format recipes for LLM prompt context."""
    context = ""
    for i, r in enumerate(recipes[:max_recipes]):
        context += f"Recipe {i+1}:\nTitle: {r.get('title')}\nIngredients: {', '.join(r.get('ingredients', []))}\nDirections: {r.get('directions', '')}\n\n"
    return context.strip()

def generate_llm_answer(query, context_recipes, model_name="models/gemini-1.5-flash"):
    """
    Use Gemini LLM to answer a user query using retrieved recipes as context.
    Loads GEMINI_API_KEY from .env if not set in environment.
    """
    # Load .env variables
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    context = format_recipes_for_context(context_recipes)
    prompt = (
        f"User question: {query}\n"
        f"Here are some relevant recipes:\n{context}\n"
        f"Based on the above, answer the user's question in detail."
    )
    response = model.generate_content(prompt)
    return response.text.strip()
