import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Try importing standard libraries
try:
    from openai import OpenAI
    from anthropic import Anthropic
except ImportError:
    st.error("Please install requirements: pip install openai anthropic scikit-learn pandas numpy")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="AI Investor Matcher (Hybrid)", layout="wide")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")
mock_mode = st.sidebar.checkbox("🛠️ Enable Mock Mode (No API Costs)", value=False)

st.sidebar.divider()

# Function to get keys from Secrets OR Sidebar
def get_api_key(key_name, display_name):
    # 1. Try to get from .streamlit/secrets.toml
    if key_name in st.secrets:
        return st.secrets[key_name]
    # 2. If not found, ask in sidebar
    else:
        return st.sidebar.text_input(f"{display_name} API Key", type="password")

if not mock_mode:
    # We look for the exact names we used in secrets.toml
    openai_api_key = get_api_key("OPENAI_API_KEY", "OpenAI")
    claude_api_key = get_api_key("ANTHROPIC_API_KEY", "Claude")
else:
    st.sidebar.warning("Mock Mode Active: Using random data.")
    openai_api_key = "mock_key"
    claude_api_key = "mock_key"
# --- Helper Functions ---

@st.cache_data
def load_investor_data(filename="investors.csv"):
    """Loads the CSV file. Creates a dummy one if it doesn't exist."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        st.warning(f"File {filename} not found. Using demo data.")
        return pd.DataFrame({
            "Investor Name": ["Demo VC 1", "Demo VC 2"],
            "Focus Area": ["Investing in AI and SaaS.", "Focus on BioTech and Health."]
        })

def get_embedding(text, client, model="text-embedding-3-small"):
    """Generates vector embeddings."""
    if mock_mode:
        return np.random.rand(1536).tolist()
    text = str(text).replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
def analyze_match_with_claude(company_desc, investor_profile, client):
    """Asks Claude for reasoning on a SPECIFIC database match."""
    if mock_mode:
        return "🤖 [MOCK] Strong match due to thesis alignment."
    
    prompt = f"""
    You are a VC analyst. 
    Startup: "{company_desc}"
    Investor Profile: "{investor_profile}"
    Task: Explain in 1 sentence why this fits, and write a 1-sentence hook for an email.
    """
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",  # <--- UPDATED THIS LINE
        max_tokens=200,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def brainstorm_additional_investors(company_desc, existing_matches, client):
    """
    Asks Claude to suggest NEW investors based on its own training data.
    """
    if mock_mode:
        return "1. **Sequoia**: Mock suggestion..."

    prompt = f"""
    You are a Senior Investment Banker.
    startup_pitch = "{company_desc}"
    We have already identified these matches: {existing_matches}
    TASK: Suggest 3-5 other top-tier investors.
    """
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",  # <--- UPDATED THIS LINE
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
# --- Main App Interface ---

st.title("🚀 Hybrid AI Investor Matcher")
st.markdown("Find investors from your **Database** AND get global suggestions from **Claude's AI Knowledge**.")

# 1. Load Data
df = load_investor_data("investors.csv")

# Initialize Clients
client_openai = None if mock_mode else OpenAI(api_key=openai_api_key)
client_claude = None if mock_mode else Anthropic(api_key=claude_api_key)

# 2. Process Database Embeddings (Cached)
if 'db_embeddings' not in st.session_state:
    st.session_state['db_embeddings'] = []

if df is not None and not df.empty:
    if not st.session_state['db_embeddings']:
        if mock_mode or openai_api_key:
            with st.spinner("Indexing Investor Database..."):
                embeddings = []
                prog_bar = st.progress(0)
                for i, row in df.iterrows():
                    emb = get_embedding(row['Focus Area'], client_openai)
                    embeddings.append(emb)
                    prog_bar.progress((i + 1) / len(df))
                st.session_state['db_embeddings'] = embeddings
                st.success(f"Indexed {len(df)} investors from CSV.")

# 3. User Input & Matching
st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    company_pitch = st.text_area("Startup Pitch:", height=150, 
        placeholder="e.g., We are building AI for automated dental scheduling...")
with col2:
    st.write("Settings")
    top_k = st.slider("Database matches:", 1, 5, 3)
    analyze_btn = st.button("Find Investors", type="primary")

if analyze_btn:
    if not company_pitch:
        st.warning("Please enter a startup description.")
    elif not st.session_state['db_embeddings']:
        st.error("Database not indexed. Check API keys.")
    else:
        # --- PART 1: Database Search (Vectors) ---
        with st.spinner("Searching internal database..."):
            pitch_embedding = get_embedding(company_pitch, client_openai)
            
            pitch_vec = np.array(pitch_embedding).reshape(1, -1)
            investor_vecs = np.array(st.session_state['db_embeddings'])
            scores = cosine_similarity(pitch_vec, investor_vecs)[0]
            
            results_df = df.copy()
            results_df['similarity_score'] = scores
            top_results = results_df.sort_values(by='similarity_score', ascending=False).head(top_k)
            
            # Save names for the exclusion list
            found_investor_names = top_results['Investor Name'].tolist()

        # --- PART 2: Display Database Matches ---
        st.subheader("📂 Matches from Your Database (CSV)")
        
        for index, row in top_results.iterrows():
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {row['Investor Name']}")
                    st.caption(row['Focus Area'])
                with c2:
                    st.metric(label="Match Score", value=f"{row['similarity_score']:.2f}")
                
                # Contextual Analysis
                if mock_mode or claude_api_key:
                    with st.spinner("Analyzing fit..."):
                        analysis = analyze_match_with_claude(company_pitch, row['Focus Area'], client_claude)
                        st.info(analysis)
                st.markdown("---")

        # --- PART 3: AI Brainstorming (Generative) ---
        st.subheader("🧠 AI Suggested Investors (Global/External)")
        st.markdown("*These investors were NOT in your CSV but were suggested by AI based on your pitch.*")
        
        if mock_mode or claude_api_key:
            with st.spinner("Brainstorming additional investors..."):
                # We pass the 'found_investor_names' to Claude so it doesn't repeat them
                ai_suggestions = brainstorm_additional_investors(company_pitch, found_investor_names, client_claude)
                st.success(ai_suggestions)
        else:
            st.warning("Enter Claude API Key to see AI suggestions.")