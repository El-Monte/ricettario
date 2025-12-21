import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
import random  # <--- Added for "Surprise Me"
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Config
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="La Cucina di Mamma", 
    page_icon="🍝",
    layout="centered",
    initial_sidebar_state="collapsed" # Hide sidebar to make it cleaner on phone
)

# --- MEMORY ---
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None

# 2. Load Data
@st.cache_data
def load_data():
    if not os.path.exists('recipes_embeddings.pkl'):
        return pd.DataFrame()
    return pd.read_pickle('recipes_embeddings.pkl')

df_original = load_data()

# 3. SMART HYBRID SEARCH (Vector + Keyword Check)
def search_recipes(dataframe, user_query):
    if dataframe.empty: return None
    
    # A. Handle "Surprise Me"
    if user_query == "SURPRISE_ME":
        return dataframe.sample(1).iloc[0]

    # B. Get Vector Embeddings (The "Meaning" Search)
    model = "models/text-embedding-004"
    query_embedding = None
    for attempt in range(3):
        try:
            query_embedding = genai.embed_content(model=model, content=user_query)['embedding']
            break
        except:
            time.sleep(1)
    
    if query_embedding is None: return None

    # C. Calculate Vector Scores
    data_embeddings = np.stack(dataframe['embedding'].values)
    vector_scores = cosine_similarity([query_embedding], data_embeddings)[0]
    
    # --- D. THE FIX: Keyword Boosting ---
    # We create a copy so we don't mess up the original data
    candidates = dataframe.copy()
    candidates['vector_score'] = vector_scores
    
    # 1. Split user query into words (e.g., "uova", "zucchine")
    user_words = user_query.lower().split()
    
    # 2. Count how many user words appear in the Ingredients column
    def count_matches(row):
        ingredients = row['Ingredients'].lower()
        title = row['Title'].lower()
        score = 0
        for word in user_words:
            # We remove the last letter to handle plurals roughly (uova/uovo)
            # e.g., "zucchine" -> "zucchin" matches "zucchina"
            clean_word = word[:-1] if len(word) > 3 else word
            
            if clean_word in ingredients or clean_word in title:
                score += 1 
        return score

    candidates['keyword_score'] = candidates.apply(count_matches, axis=1)
    
    # E. Final Sort
    # Priority 1: Who has the most matching keywords?
    # Priority 2: Who has the best vector math score?
    best_matches = candidates.sort_values(
        by=['keyword_score', 'vector_score'], 
        ascending=[False, False]
    )
    
    # Return the absolute winner
    return best_matches.iloc[0]

# --- 🎨 THE INTERFACE ---

st.markdown("<h1 style='text-align: center; color: #E63946;'>🍝 La Cucina di Mamma</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Ricette buone, sane e veloci.</p>", unsafe_allow_html=True)

# --- QUICK TAGS (NEW FEATURE) ---
# These buttons act like shortcuts
st.write("")
tags_cols = st.columns(4)
query_from_tags = None

if tags_cols[0].button("🥦 Veg", use_container_width=True):
    query_from_tags = "verdure vegetariano leggero"
if tags_cols[1].button("⚡ Veloce", use_container_width=True):
    query_from_tags = "veloce facile rapido"
if tags_cols[2].button("🍖 Carne", use_container_width=True):
    query_from_tags = "carne pollo manzo"
if tags_cols[3].button("🎲 Sorpresa", use_container_width=True):
    query_from_tags = "SURPRISE_ME"

# Sidebar
with st.sidebar:
    st.header("⚙️ Impostazioni")
    max_time = st.slider("Tempo max (min)", 10, 180, 120)

if not df_original.empty:
    df_filtered = df_original[df_original['Time'] <= max_time]
else:
    df_filtered = pd.DataFrame()

# --- SEARCH BAR (Fixed with Form) ---
# We use a Form so "Enter" key works automatically
with st.form("search_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        # We add 'key' to ensure unique ID
        user_input_text = st.text_input("Ingrediente:", placeholder="Es: zucchine...", label_visibility="collapsed")
    with col2:
        # This button now reacts to the Enter key too
        search_pressed = st.form_submit_button("🔍", type="primary")

# LOGIC: Determine what to search for
final_query = None

# 1. Did she use the Form (Type + Enter/Click)?
if search_pressed and user_input_text:
    final_query = user_input_text

# 2. OR Did she click a Quick Tag?
elif query_from_tags:
    final_query = query_from_tags

# --- RUN SEARCH ---
if final_query:
    with st.spinner("Cerco l'ispirazione..."):
        best_recipe = search_recipes(df_filtered, final_query)
        if best_recipe is not None:
            st.session_state.selected_recipe = best_recipe
            st.session_state.history = [] # Reset chat
        else:
            st.error("Nessuna ricetta trovata!")

# --- DISPLAY RESULT ---
if st.session_state.selected_recipe is not None:
    rec = st.session_state.selected_recipe
    
    st.markdown("---")
    
    # Title Card
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
        <h2 style="color: #2A9D8F; margin:0;">{rec['Title']}</h2>
        <p style="margin-top: 5px;">⏱️ <b>{rec['Time']} min</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # Tab Layout (Cleaner than columns)
    tab1, tab2, tab3 = st.tabs(["📝 Ricetta", "🛒 Lista Spesa", "💬 Chiedi allo Chef"])
    
    with tab1:
        # Ingredients & Steps
        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            st.markdown("#### Ingredienti")
            ingredients_list = rec['Ingredients'].split(',')
            for ing in ingredients_list:
                st.markdown(f"▫️ {ing.strip()}")
        with col_b:
            st.markdown("#### Preparazione")
            st.write(rec['Instructions'])

    with tab2:
        # SHOPPING LIST HELPER (NEW)
        st.info("Copia questa lista per WhatsApp!")
        
        # Use AI to format it as a checkbox list
        if st.button("Genera Lista Checklist"):
            with st.spinner("Scrivo la lista..."):
                prompt_list = f"Trasforma questi ingredienti in una lista della spesa con checkbox per WhatsApp (senza testo introduttivo): {rec['Ingredients']}"
                checklist = ask_ai(prompt_list)
                st.code(checklist, language="markdown")

    with tab3:
        # Chat
        st.caption("Dubbi? Chiedi qui.")
        chat_cont = st.container(height=300) # Scrollable container
        
        for message in st.session_state.history:
            with chat_cont.chat_message(message["role"]):
                st.markdown(message["content"])
            
        if prompt := st.chat_input("Es: Posso surgelarla?"):
            st.session_state.history.append({"role": "user", "content": prompt})
            with chat_cont.chat_message("user"):
                st.markdown(prompt)
            
            with chat_cont.chat_message("assistant"):
                with st.spinner("..."):
                    context = f"Ricetta: {rec['Title']}. Istruzioni: {rec['Instructions']}."
                    full_prompt = f"Contesto: {context}\nDomanda: {prompt}\nRispondi brevemente."
                    response = ask_ai(full_prompt)
                    st.markdown(response)
            st.session_state.history.append({"role": "assistant", "content": response})