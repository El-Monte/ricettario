import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Config & Setup
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Le Ricette della Mamma", page_icon="🍝")

# --- MEMORY SETUP (Session State) ---
# This "backpack" holds data so we don't lose it
if "history" not in st.session_state:
    st.session_state.history = [] # Stores the chat conversation
if "current_recipes" not in st.session_state:
    st.session_state.current_recipes = None # Stores the recipes found
if "context_text" not in st.session_state:
    st.session_state.context_text = "" # Stores the text we sent to the AI

# 2. Load Data (Cached for speed)
@st.cache_data
def load_data():
    if not os.path.exists('recipes_embeddings.pkl'):
        st.error("File embeddings non trovato. Esegui create_embeddings.py!")
        return pd.DataFrame()
    return pd.read_pickle('recipes_embeddings.pkl')

df_original = load_data()

# 3. Search Logic
def find_top_recipes(dataframe, user_query, top_k=6):
    if dataframe.empty: return dataframe
    
    # Generate embedding for the query
    model = "models/text-embedding-004"
    query_embedding = genai.embed_content(model=model, content=user_query)['embedding']
    
    # Calculate similarity
    data_embeddings = np.stack(dataframe['embedding'].values)
    scores = cosine_similarity([query_embedding], data_embeddings)[0]
    
    # Get top results
    real_k = min(top_k, len(dataframe))
    top_indices = scores.argsort()[-real_k:][::-1]
    return dataframe.iloc[top_indices]

# 4. AI Generation Logic (The Chef)
def talk_to_chef(prompt_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt_text)
    return response.text

# --- MAIN INTERFACE ---
st.title("🍝 Le Ricette della Mamma")

# Sidebar Filters
st.sidebar.header("Filtri")
max_time = st.sidebar.slider("Tempo massimo (minuti)", 10, 120, 120)

# Filter Data
if not df_original.empty:
    df_filtered = df_original[df_original['Time'] <= max_time]
else:
    df_filtered = pd.DataFrame()

# --- SECTION 1: SEARCH (The RAG Part) ---
st.markdown("### 1. Cerca una ricetta")
user_ingredients = st.text_input("Cosa hai nel frigo?", placeholder="Es: zucchine, uova, tofu...")

if st.button("Trova Ricetta"):
    if user_ingredients:
        with st.spinner("Consulto il ricettario..."):
            # A. Search
            top_recipes = find_top_recipes(df_filtered, user_ingredients)
            
            if top_recipes.empty:
                st.warning("Nessuna ricetta trovata con questi filtri.")
            else:
                # B. Save Context to Memory (So the Chat knows what we are talking about)
                st.session_state.current_recipes = top_recipes
                
                # Create a string representation of the recipes for the AI
                context_str = ""
                for index, row in top_recipes.iterrows():
                    context_str += f"- {row['Title']} ({row['Time']} min): {row['Ingredients']} (Istruzioni: {row['Instructions']})\n"
                st.session_state.context_text = context_str
                
                # C. Initial Prompt to the Chef
                initial_prompt = f"""
                Sei uno chef italiano esperto e creativo.
                L'utente ha questi ingredienti: "{user_ingredients}".
                
                Ecco le ricette che ho trovato nel database (potrebbero non essere tutte pertinenti):
                {context_str}
                
                IL TUO COMPITO:
                1. Analizza gli ingredienti dell'utente. Stanno bene insieme?
                2. SE STANNO BENE INSIEME: Scegli la ricetta migliore dal database e spiegala.
                3. SE NON STANNO BENE INSIEME (es. "cioccolato e tonno"):
                   - Non forzare un mix disgustoso!
                   - Proponi DUE ricette separate (una per il primo ingrediente, una per il secondo).
                   - OPPURE, se sei creativo, proponi un abbinamento "gourmet" inaspettato ma avverti l'utente che è un esperimento.
                
                Usa un tono simpatico e dai del "tu".
                Finisci con il dire: "Buon appetito Castagnetta!"
                """
                
                # D. Generate and Save to History
                response_text = talk_to_chef(initial_prompt)
                
                # Reset history for a new search
                st.session_state.history = []
                st.session_state.history.append({"role": "assistant", "content": response_text})

# --- SECTION 2: DISPLAY & CHAT (The Memory Part) ---
# If we have a search result in memory, show it
if st.session_state.current_recipes is not None:
    
    # 1. Show the "Best Match" Table
    best = st.session_state.current_recipes.iloc[0]
    st.info(f"💡 Suggerimento Top: **{best['Title']}** ({best['Time']} min)")
    
    # 2. Show Chat History
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 3. User Chat Input (Follow-up questions)
    if prompt := st.chat_input("Dubbi? Chiedi allo chef (Es: posso usare il burro?)"):
        
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Lo chef risponde..."):
                # We send the previous context + the new question
                full_prompt = f"""
                Contesto Ricette:
                {st.session_state.context_text}
                
                Domanda dell'utente: "{prompt}"
                
                Rispondi basandoti SULLA RICETTA di cui stiamo parlando. Sii breve.
                """
                
                response = talk_to_chef(full_prompt)
                st.markdown(response)
                
        # Add assistant response to history
        st.session_state.history.append({"role": "assistant", "content": response})