import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
import re # Added for better text cleaning
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
    initial_sidebar_state="collapsed"
)

# --- MEMORY ---
if "history" not in st.session_state:
    st.session_state.history = []
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "search" # 'search' or 'recipe_detail'
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# 2. Load Data
@st.cache_data
def load_data():
    if not os.path.exists('recipes_embeddings.pkl'):
        return pd.DataFrame()
    return pd.read_pickle('recipes_embeddings.pkl')

df_original = load_data()

# 3. ROBUST SEARCH (Returns Top 3)
def search_recipes(dataframe, user_query):
    if dataframe.empty: return pd.DataFrame()
    
    # Handle "Surprise Me"
    if user_query == "SURPRISE_ME":
        return dataframe.sample(3) # Return 3 random ones

    # Vector Search
    model = "models/text-embedding-004"
    query_embedding = None
    for attempt in range(3):
        try:
            query_embedding = genai.embed_content(model=model, content=user_query)['embedding']
            break
        except:
            time.sleep(1)
            
    if query_embedding is None: return pd.DataFrame()

    data_embeddings = np.stack(dataframe['embedding'].values)
    vector_scores = cosine_similarity([query_embedding], data_embeddings)[0]
    
    # Hybrid Scoring
    candidates = dataframe.copy()
    candidates['vector_score'] = vector_scores
    
    # Better text cleaning
    # Remove commas and common Italian stop words
    clean_query = re.sub(r'[^\w\s]', '', user_query.lower()) # remove punctuation
    stop_words = ["e", "con", "il", "di", "la", "ho", "delle"]
    user_words = [w for w in clean_query.split() if w not in stop_words]
    
    def count_matches(row):
        ingredients = row['Ingredients'].lower()
        title = row['Title'].lower()
        score = 0
        for word in user_words:
            # Check if the word (or singular version) exists
            if word in ingredients or word in title:
                score += 1
            elif len(word) > 4 and word[:-1] in ingredients: # simple plural check
                score += 1
        return score

    candidates['keyword_score'] = candidates.apply(count_matches, axis=1)
    
    # Sort: Keywords first, then Context
    best_matches = candidates.sort_values(
        by=['keyword_score', 'vector_score'], 
        ascending=[False, False]
    )
    
    # Return TOP 3
    return best_matches.head(3)

# 4. AI Logic
def ask_ai(prompt_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt_text)
    return response.text

# --- 🎨 INTERFACE ---

# Header
if st.session_state.view_mode == "search":
    st.markdown("<h1 style='text-align: center; color: #E63946;'>🍝 La Cucina di Mamma</h1>", unsafe_allow_html=True)
else:
    # Smaller header when looking at a recipe
    if st.button("⬅️ Torna alla ricerca"):
        st.session_state.view_mode = "search"
        st.session_state.selected_recipe = None
        st.rerun()

# --- VIEW 1: SEARCH SCREEN ---
if st.session_state.view_mode == "search":
    
    # Quick Tags
    tags_cols = st.columns(4)
    query_from_tags = None
    if tags_cols[0].button("🥦 Veg", use_container_width=True): query_from_tags = "verdure vegetariano"
    if tags_cols[1].button("⚡ Veloce", use_container_width=True): query_from_tags = "veloce facile"
    if tags_cols[2].button("🍖 Carne", use_container_width=True): query_from_tags = "carne pollo manzo"
    if tags_cols[3].button("🎲 Sorpresa", use_container_width=True): query_from_tags = "SURPRISE_ME"

    # Search Bar
    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input_text = st.text_input("Ingrediente:", placeholder="Es: zucchine...", label_visibility="collapsed")
        with col2:
            search_pressed = st.form_submit_button("🔍", type="primary")

    # Sidebar Filter
    with st.sidebar:
        st.header("⚙️ Impostazioni")
        max_time = st.slider("Tempo max (min)", 10, 180, 120)

    # Determine Query
    final_query = None
    if search_pressed and user_input_text:
        final_query = user_input_text
    elif query_from_tags:
        final_query = query_from_tags

    # EXECUTE SEARCH
    if final_query:
        # Filter by time first
        if not df_original.empty:
            df_filtered = df_original[df_original['Time'] <= max_time]
        else:
            df_filtered = pd.DataFrame()
            
        with st.spinner("Cerco le opzioni migliori..."):
            results = search_recipes(df_filtered, final_query)
            
            if not results.empty:
                st.session_state.search_results = results
            else:
                st.error("Nessuna ricetta trovata!")

    # DISPLAY RESULTS (Top 3 Cards)
    if st.session_state.search_results is not None:
        st.write("")
        st.subheader("Ecco cosa ho trovato:")
        
        results = st.session_state.search_results
        
        # Loop through the top 3 results
        for index, row in results.iterrows():
            # Create a clickable card look using a button
            # We use a container to make it look nice
            with st.container(border=True):
                col_text, col_btn = st.columns([3, 1])
                with col_text:
                    st.markdown(f"**{row['Title']}**")
                    st.caption(f"⏱️ {row['Time']} min | 🛒 {row['Ingredients'][:50]}...")
                with col_btn:
                    # Unique key is needed for buttons in loops
                    if st.button("Vedi", key=f"btn_{index}", use_container_width=True):
                        st.session_state.selected_recipe = row
                        st.session_state.view_mode = "recipe_detail"
                        st.session_state.history = [] # Reset chat
                        st.rerun()

# --- VIEW 2: RECIPE DETAIL ---
elif st.session_state.view_mode == "recipe_detail" and st.session_state.selected_recipe is not None:
    rec = st.session_state.selected_recipe
    
    # Title Card
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; margin-bottom: 20px;">
        <h2 style="color: #2A9D8F; margin:0;">{rec['Title']}</h2>
        <p style="margin-top: 5px;">⏱️ <b>{rec['Time']} min</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Ricetta", "🛒 Spesa", "💬 Chef"])
    
    with tab1:
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
        st.info("Lista per WhatsApp:")
        if st.button("Crea Checklist"):
            with st.spinner("Genero..."):
                checklist = ask_ai(f"Lista spesa checkbox per: {rec['Ingredients']}")
                st.code(checklist, language="markdown")

    with tab3:
        chat_cont = st.container(height=300)
        for msg in st.session_state.history:
            with chat_cont.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Chiedi consiglio..."):
            st.session_state.history.append({"role": "user", "content": prompt})
            with chat_cont.chat_message("user"):
                st.markdown(prompt)
            
            with chat_cont.chat_message("assistant"):
                with st.spinner("..."):
                    ctx = f"Ricetta: {rec['Title']}. Istruzioni: {rec['Instructions']}."
                    resp = ask_ai(f"Contesto: {ctx}\nDomanda: {prompt}\nRispondi brevemente.")
                    st.markdown(resp)
            st.session_state.history.append({"role": "assistant", "content": resp})