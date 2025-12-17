import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Config
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="La Cucina di Mamma", 
    page_icon="🍝",
    layout="centered"
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

# 3. Search Logic
def search_recipes(dataframe, user_query):
    if dataframe.empty: return None
    
    model = "models/text-embedding-004"
    query_embedding = None
    
    for attempt in range(3):
        try:
            query_embedding = genai.embed_content(model=model, content=user_query)['embedding']
            break
        except:
            time.sleep(1)
            
    if query_embedding is None: return None

    data_embeddings = np.stack(dataframe['embedding'].values)
    scores = cosine_similarity([query_embedding], data_embeddings)[0]
    
    best_index = np.argmax(scores)
    return dataframe.iloc[best_index]

# 4. AI Logic
def ask_ai(prompt_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt_text)
    return response.text

# --- 🎨 THE "HOMEY" INTERFACE ---

# Custom Title with Color (Tomato Red)
st.markdown("<h1 style='text-align: center; color: #E63946;'>🍝 La Cucina di Mamma</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Cosa cuciniamo di buono oggi?</h4>", unsafe_allow_html=True)
st.write("") # Spacer

# Sidebar Filters (Hidden slightly to keep main view clean)
with st.sidebar:
    st.header("⚙️ Impostazioni")
    max_time = st.slider("Quanto tempo hai? (min)", 10, 120, 120)
    st.caption("Filtra le ricette troppo lunghe.")

if not df_original.empty:
    df_filtered = df_original[df_original['Time'] <= max_time]
else:
    df_filtered = pd.DataFrame()

# --- SEARCH BAR (Centered & Clean) ---
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    user_ingredients = st.text_input("📝 Dimmi gli ingredienti:", placeholder="Es: ho delle zucchine e uova...")
    search_btn = st.button("🍳 Trova la Ricetta", use_container_width=True, type="primary")

if search_btn:
    if user_ingredients:
        with st.spinner("Sfoglio il quaderno delle ricette..."):
            best_recipe = search_recipes(df_filtered, user_ingredients)
            
            if best_recipe is not None:
                st.session_state.selected_recipe = best_recipe
                st.session_state.history = [] # Reset chat
            else:
                st.error("Non ho trovato nulla! Prova con altri ingredienti.")

# --- DISPLAY RESULT (The Pretty Part) ---
if st.session_state.selected_recipe is not None:
    rec = st.session_state.selected_recipe
    
    st.markdown("---") # Divider line
    
    # Recipe Header
    st.markdown(f"<h2 style='text-align: center; color: #2A9D8F;'>✨ {rec['Title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>⏱️ <b>Tempo:</b> {rec['Time']} minuti</p>", unsafe_allow_html=True)
    
    # Layout: Ingredients Left, Instructions Right
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    with col_left:
        st.info("🛒 **Ingredienti**")
        # Turn comma-separated string into a nice bullet list
        ingredients_list = rec['Ingredients'].split(',')
        for ing in ingredients_list:
            st.markdown(f"- {ing.strip()}")
            
    with col_right:
        st.success("🔥 **Preparazione**")
        st.write(rec['Instructions'])
    
    st.markdown("---")
    
    # --- CHAT BOT SECTION ---
    st.markdown("#### 👩‍🍳 L'Angolo dello Chef")
    st.caption("Hai dubbi sulla preparazione? Chiedi qui sotto!")
    
    # Chat container
    chat_container = st.container(border=True)
    with chat_container:
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
    # Input area
    if prompt := st.chat_input("Es: Posso usare il burro invece dell'olio?"):
        
        # User message
        st.session_state.history.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI Reply
            with st.chat_message("assistant"):
                with st.spinner("Sto scrivendo..."):
                    context = f"Ricetta: {rec['Title']}. Istruzioni: {rec['Instructions']}."
                    full_prompt = f"Contesto: {context}\nDomanda Mamma: {prompt}\nRispondi in modo gentile e breve in italiano."
                    
                    response = ask_ai(full_prompt)
                    st.markdown(response)
        
        st.session_state.history.append({"role": "assistant", "content": response})