import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# 1. Load the hidden API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Load your recipes
df = pd.read_csv('recipes.csv', sep=';')

print("⏳ Creating embeddings... (talking to Google Gemini)")

# 3. Define a helper function for Google
def get_embedding(text):
    # We use the specific embedding model from Google
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    # Google sometimes requires a tiny pause to not get overwhelmed
    time.sleep(0.5) 
    return result['embedding']

# 4. Prepare the text to search
df['search_text'] = df['Title'] + " " + df['Ingredients']

# 5. Create the embeddings
# This sends the text to Google and saves the numbers
try:
    df['embedding'] = df['search_text'].apply(get_embedding)
    
    # 6. Save the result
    df.to_pickle('recipes_embeddings.pkl')
    print("✅ Done! Created 'recipes_embeddings.pkl'. Ready to search.")
    
except Exception as e:
    print("❌ Error:", e)