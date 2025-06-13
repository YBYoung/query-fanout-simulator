import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
from datetime import datetime
import time
from typing import List, Dict, Tuple
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Query Fan-Out Simulator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Session state for model initialization
if 'sentence_model' not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dynamic CSS function
def get_css_styles(dark_mode):
    if dark_mode:
        return """
<style>
    /* Dark Mode Styles */
    .stApp { background-color: #0f0f0f !important; color: #ffffff !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { 
        color: #ffffff !important; 
    }
    
    /* Query Card Styles - Dark Mode */
    .query-card {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .query-card:hover {
        border-color: #ff8c42 !important;
        box-shadow: 0 8px 12px rgba(255, 140, 66, 0.1) !important;
    }
    .query-title { 
        color: #ffffff !important; 
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.75rem !important;
    }
    .query-label { 
        color: #999999 !important; 
        font-weight: 600 !important;
    }
    .query-type { 
        color: #66aaff !important; 
        font-weight: 600 !important;
    }
    .query-content { 
        color: #e0e0e0 !important; 
    }
    .query-reasoning { 
        color: #b0b0b0 !important; 
    }
    
    /* Form Elements - Dark Mode */
    .stSelectbox > div > div { 
        background-color: #1a1a1a !important; 
        color: #ffffff !important; 
        border: 1px solid #333333 !important;
    }
    .stTextInput > div > div > input { 
        background-color: #1a1a1a !important; 
        color: #ffffff !important; 
        border: 1px solid #333333 !important;
    }
    .stTextInput > div > div > input::placeholder { 
        color: #666666 !important;
    }
    
    /* Buttons - Dark Mode */
    .stButton > button { 
        background-color: #ff8c42 !important; 
        color: #000000 !important; 
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover { 
        background-color: #ff7a2e !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255, 140, 66, 0.3) !important;
    }
    
    /* Tabs - Dark Mode */
    .stTabs [data-baseweb="tab-list"] { 
        background-color: transparent !important;
        border-bottom: 2px solid #333333 !important;
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent !important; 
        color: #999999 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        margin-right: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: transparent !important; 
        color: #ff8c42 !important;
        border-bottom: 3px solid #ff8c42 !important;
        border-radius: 0 !important;
    }
    
    /* Other Elements - Dark Mode */
    .stExpander { 
        background-color: #1a1a1a !important; 
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    .stAlert > div { 
        background-color: #1a2744 !important; 
        color: #ffffff !important;
        border: 1px solid #2a3f5f !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button { 
        background-color: #1a1a1a !important; 
        color: #ffffff !important; 
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover { 
        background-color: #2a2a2a !important; 
        border-color: #ff8c42 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Metrics - Dark Mode */
    [data-testid="metric-container"] {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ff8c42 !important;
    }
    
    /* Typography */
    html, body, [class*="css"] { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
        font-size: 16px; 
        line-height: 1.6; 
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
</style>
        """
    else:
        return """
<style>
    /* Light Mode Styles */
    .stApp { background-color: #ffffff !important; color: #1a1a1a !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { 
        color: #1a1a1a !important; 
    }
    
    /* Query Card Styles - Light Mode */
    .query-card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.3s ease !important;
    }
    .query-card:hover {
        border-color: #dc6b2f !important;
        box-shadow: 0 4px 8px rgba(220, 107, 47, 0.1) !important;
    }
    .query-title { 
        color: #1a1a1a !important; 
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.75rem !important;
    }
    .query-label { 
        color: #666666 !important; 
        font-weight: 600 !important;
    }
    .query-type { 
        color: #0066cc !important; 
        font-weight: 600 !important;
    }
    .query-content { 
        color: #333333 !important; 
    }
    .query-reasoning { 
        color: #555555 !important; 
    }
    
    /* Form Elements - Light Mode */
    .stSelectbox > div > div { 
        background-color: #ffffff !important; 
        color: #1a1a1a !important; 
        border: 1px solid #e0e0e0 !important;
    }
    .stTextInput > div > div > input { 
        background-color: #ffffff !important; 
        color: #1a1a1a !important; 
        border: 1px solid #e0e0e0 !important;
    }
    .stTextInput > div > div > input::placeholder { 
        color: #999999 !important;
    }
    
    /* Buttons - Light Mode */
    .stButton > button { 
        background-color: #dc6b2f !important; 
        color: #ffffff !important; 
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover { 
        background-color: #c55a24 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(220, 107, 47, 0.3) !important;
    }
    
    /* Tabs - Light Mode */
    .stTabs [data-baseweb="tab-list"] { 
        background-color: transparent !important;
        border-bottom: 2px solid #e0e0e0 !important;
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent !important; 
        color: #666666 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        margin-right: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        color: #1a1a1a !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: transparent !important; 
        color: #dc6b2f !important;
        border-bottom: 3px solid #dc6b2f !important;
        border-radius: 0 !important;
    }
    
    /* Other Elements - Light Mode */
    .stExpander { 
        background-color: #fafafa !important; 
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    .stAlert > div { 
        background-color: #e7f3ff !important; 
        color: #1a1a1a !important;
        border: 1px solid #b3d9ff !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button { 
        background-color: #ffffff !important; 
        color: #1a1a1a !important; 
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover { 
        background-color: #f5f5f5 !important; 
        border-color: #dc6b2f !important;
        transform: translateY(-1px) !important;
    }
    
    /* Metrics - Light Mode */
    [data-testid="metric-container"] {
        background-color: #fafafa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #dc6b2f !important;
    }
    
    /* Typography */
    html, body, [class*="css"] { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
        font-size: 16px; 
        line-height: 1.6; 
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
</style>
        """
        
# Apply the CSS
st.markdown(get_css_styles(st.session_state.dark_mode), unsafe_allow_html=True)

# Session state for view
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'card'

# API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

QUERY_TYPE_DESCRIPTIONS = {
    "reformulation": {
        "name": "Reformulation",
        "description": "Alternative phrasings of the original query that maintain the same core intent.",
        "example": "For 'best pizza NYC' → 'top rated pizza restaurants in New York'",
        "strategy": "Helps capture different ways users might search for the same information."
    },
    "related": {
        "name": "Related",
        "description": "Semantically adjacent queries that explore similar topics or concepts.",
        "example": "For 'best pizza NYC' → 'Italian restaurants Manhattan'",
        "strategy": "Captures users interested in closely related information."
    },
    "implicit": {
        "name": "Implicit",
        "description": "Queries addressing unstated user needs or underlying intentions.",
        "example": "For 'best pizza NYC' → 'affordable late night food NYC'",
        "strategy": "Surfaces content for what users actually want but didn't explicitly ask."
    },
    "comparative": {
        "name": "Comparative",
        "description": "Queries that compare options, products, or solutions.",
        "example": "For 'best pizza NYC' → 'Dominos vs Papa Johns NYC'",
        "strategy": "Essential for decision-making content and comparison guides."
    },
    "entity_expansion": {
        "name": "Entity Expansion",
        "description": "Queries that broaden or narrow scope using specific brands/features.",
        "example": "For 'best pizza NYC' → 'Prince Street Pizza reviews'",
        "strategy": "Captures long-tail variations with specific entities."
    },
    "personalized": {
        "name": "Personalized",
        "description": "Location or context-specific query variations.",
        "example": "For 'best pizza NYC' → 'pizza delivery near me Manhattan'",
        "strategy": "Uses templates like '[near me]' to capture local intent."
    }
}

def clean_gemini_response(raw: str) -> str:
    return raw.strip().removeprefix("```json").removesuffix("```").strip()

def parse_gemini_response(response_text: str) -> Tuple[str, int, List[Dict]]:
    try:
        cleaned_text = clean_gemini_response(response_text)
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            reasoning = data.get("reasoning_steps", "")
            target_count = data.get("target_query_count", 0)
            queries = data.get("queries", [])
            return reasoning, target_count, queries
    except Exception as e:
        st.warning(f"Failed to parse JSON response: {e}")
    return "Could not extract structured reasoning.", 0, []

def deduplicate_queries(queries: List[str], threshold: float = 0.85) -> List[int]:
    """
    Return indices of queries to keep after deduplication.
    Uses the cached sentence transformer model.
    """
    if not queries:
        return []
    
    try:
        # Use the cached model from session state
        model_emb = st.session_state.sentence_model
        
        # Encode queries
        embeddings = model_emb.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        
        keep_indices = []
        for idx, embedding in enumerate(embeddings):
            if not keep_indices:
                keep_indices.append(idx)
            else:
                # Compare with already kept embeddings
                kept_embeddings = embeddings[keep_indices]
                similarities = util.pytorch_cos_sim(embedding, kept_embeddings)
                
                # Keep if all similarities are below threshold
                if torch.max(similarities).item() < threshold:
                    keep_indices.append(idx)
        
        return keep_indices
        
    except Exception as e:
        st.warning(f"Deduplication skipped due to error: {str(e)}")
        # Return all indices if deduplication fails
        return list(range(len(queries)))

def generate_synthetic_queries(query: str) -> Tuple[str, int, pd.DataFrame]:
    try:
        prompt = f"""You are an expert at understanding how Google's AI Mode generates synthetic queries through query fan-out.

Given the user query: "{query}"

First, analyze the complexity, breadth, and ambiguity of this query.
Based on this, decide how many synthetic queries are required for comprehensive coverage (typically between 14 and 30, with more for broad/ambiguous queries, fewer for narrow/specific ones).
State the number in "target_query_count" and generate up to that number.

Include as many queries as possible that are:
- Distinct in their wording or underlying intent (it's okay if some overlap, as long as they would return different results in search)
- Relevant and useful to the core topic (avoid queries that are totally off-topic, nonsensical, or empty)
- Covering different perspectives, subtopics, or intent angles—even if some are less common
- Meet the needs of different user intents or coverage areas
- High-quality (not vague, not just a copy of the seed, not empty, not generic)

Distribute your queries across these categories as appropriate:
- Reformulation: Different phrasings, same intent
- Related: Semantically adjacent topics
- Implicit: Unstated user needs
- Comparative: Comparing options/solutions
- Entity expansion: Specific brands/features/narrower scope
- Personalized: Location/context specific

For each query, provide:
1. The query text
2. The query type (reformulation, related, implicit, comparative, entity_expansion, or personalized)
3. The user intent behind this query
4. Your reasoning for why this query was selected (2-3 sentences)

Respond in this exact JSON format:
{{
    "reasoning_steps": "Your analysis of the query complexity and why you chose to generate X number of queries...",
    "target_query_count": [number you decided to generate, usually between 14-30],
    "queries": [
        {{
            "query": "the synthetic query text",
            "type": "query_type",
            "user_intent": "what the user wants to achieve",
            "reasoning": "why this query was selected"
        }}
    ]
}}"""
        response = model.generate_content(prompt)
        reasoning, target_count, queries_data = parse_gemini_response(response.text)

        if not target_count or not isinstance(target_count, int) or target_count < 8:
            target_count = 18

        if not queries_data:
            return "Failed to parse queries", 0, pd.DataFrame()

        df = pd.DataFrame(queries_data)
        for col in ['query', 'type', 'user_intent', 'reasoning']:
            if col not in df.columns:
                df[col] = ''

        # Initial quality filter
        def is_low_quality(q):
            return (
                pd.isna(q)
                or len(q.strip()) < 4
                or q.strip().lower() == query.strip().lower()
                or q.strip().lower() in ['n/a', 'none', 'no idea', '']
            )

        df = df[~df['query'].apply(is_low_quality)].copy()

        # Embedding-based deduplication
        query_list = df['query'].tolist()
        if query_list:
            keep_indices = deduplicate_queries(query_list, threshold=0.85)
            df = df.iloc[keep_indices].reset_index(drop=True)

        return reasoning, target_count, df

    except Exception as e:
        st.error(f"Error generating or filtering queries: {str(e)}")
        return f"Error: {str(e)}", 0, pd.DataFrame()

def process_batch_queries(queries: List[str]) -> pd.DataFrame:
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, query in enumerate(queries):
        status_text.text(f"Processing: {query}")
        reasoning, target_count, df = generate_synthetic_queries(query)
        if not df.empty:
            df['original_query'] = query
            df['reasoning_steps'] = reasoning
            df['target_count'] = target_count
            df['actual_count'] = len(df)
            all_results.append(df)
        progress_bar.progress((i + 1) / len(queries))
        if i < len(queries) - 1:
            time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def display_query_results(df: pd.DataFrame, reasoning: str, target_count: int, view_mode: str = 'card'):
    # Generation statistics
    st.markdown("### Generation Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Queries", target_count)
    with col2:
        st.metric("Generated", len(df))
    with col3:
        st.metric("Success Rate", f"{(len(df)/target_count*100):.1f}%")
    
    st.markdown("### AI's Reasoning")
    st.info(reasoning)

    # Generated queries
    st.markdown("### Generated Synthetic Queries")
    for idx, row in df.iterrows():
        st.markdown(f"""
        <div class="query-card">
            <div class="query-title">{row['query']}</div>
            <div style="margin-bottom: 0.5rem;">
                <span class="query-label">Type:</span>
                <span class="query-type">{row['type'].replace('_', ' ').title()}</span>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span class="query-label">Intent:</span>
                <span class="query-content">{row['user_intent']}</span>
            </div>
            <div>
                <span class="query-label">Reasoning:</span>
                <span class="query-reasoning">{row['reasoning']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 View as Table", expanded=False):
        st.dataframe(
            df[['query', 'type', 'user_intent', 'reasoning']],
            use_container_width=True,
            height=400
        )

# Sidebar with dark mode toggle
with st.sidebar:
    new_dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark_mode
        st.rerun()

# Header
st.title("Query Fan-Out Simulator")
st.markdown("Understand how AI search systems expand queries to capture user intent")

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Upload", "How It Works"])

with tab1:
    st.markdown("### Analyze a single search query")
    st.markdown("Enter a query to see how AI search systems might expand it into related searches.")
    single_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., best pediatric dentist, affordable car insurance, learn python programming",
        help="The more specific your query, the better the results"
    )
    if st.button("Generate Synthetic Queries", type="primary", use_container_width=True):
        if single_query:
            with st.spinner("AI is analyzing your query..."):
                reasoning, target_count, df = generate_synthetic_queries(single_query)
                if not df.empty:
                    display_query_results(df, reasoning, target_count, st.session_state.view_mode)
                    # Download button
                    st.markdown("---")
                    csv = df.to_csv(index=False)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv,
                            file_name=f"query_fanout_{single_query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        else:
            st.warning("Please enter a query to analyze")

with tab2:
    st.markdown("### Batch process multiple queries")
    st.markdown("Upload a CSV file with multiple queries to analyze them all at once.")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should contain a column with your queries"
    )
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(input_df)} rows from file")
            with st.expander("Preview uploaded data"):
                st.dataframe(input_df.head())
            if not input_df.empty:
                query_column = st.selectbox(
                    "Select query column:",
                    input_df.columns,
                    help="Choose the column that contains your search queries"
                )
                st.markdown("**Sample queries from your file:**")
                sample_queries = input_df[query_column].dropna().head(3).tolist()
                for q in sample_queries:
                    st.caption(f"• {q}")
                if st.button("Process All Queries", type="primary", use_container_width=True):
                    queries = input_df[query_column].dropna().tolist()
                    if queries:
                        st.markdown(f"### Processing {len(queries)} queries...")
                        results_df = process_batch_queries(queries)
                        if not results_df.empty:
                            st.balloons()
                            st.success(f"Generated {len(results_df)} synthetic queries!")
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Queries", len(results_df))
                            with col2:
                                st.metric("Original Queries", len(queries))
                            with col3:
                                total_target = results_df.groupby('original_query')['target_count'].first().sum()
                                st.metric("Target Total", total_target)
                            with col4:
                                st.metric("Success Rate", f"{(len(results_df)/total_target*100):.1f}%")
                            # Show distribution
                            st.markdown("### Query Type Distribution")
                            type_counts = results_df['type'].value_counts()
                            chart_data = pd.DataFrame({
                                'Query Type': [QUERY_TYPE_DESCRIPTIONS.get(t, {}).get("name", t) for t in type_counts.index],
                                'Count': type_counts.values
                            })
                            st.bar_chart(chart_data.set_index('Query Type'))
                            # Show results grouped by original query
                            st.markdown("### Results by Original Query")
                            for orig_query in results_df['original_query'].unique()[:10]:
                                query_results = results_df[results_df['original_query'] == orig_query]
                                target = query_results.iloc[0]['target_count']
                                actual = query_results.iloc[0]['actual_count']
                                with st.expander(f"📝 {orig_query} (Generated: {actual}/{target})", expanded=False):
                                    st.markdown(f"**Success Rate:** {(actual/target*100):.1f}%")
                                    for _, row in query_results.iterrows():
                                        st.markdown(f"""
                                        <div class="query-card">
                                            <div class="query-title">{row['query']}</div>
                                            <div style="margin-bottom: 0.5rem;">
                                                <span class="query-label">Type:</span>
                                                <span class="query-type">{row['type'].replace('_', ' ').title()}</span>
                                            </div>
                                            <div style="margin-bottom: 0.5rem;">
                                                <span class="query-label">Intent:</span>
                                                <span class="query-content">{row['user_intent']}</span>
                                            </div>
                                            <div>
                                                <span class="query-label">Reasoning:</span>
                                                <span class="query-reasoning">{row['reasoning']}</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with st.expander("View all queries as table", expanded=False):
                                        st.dataframe(query_results[['query', 'type', 'user_intent', 'reasoning']])
                            # Download button
                            st.markdown("---")
                            csv = results_df.to_csv(index=False)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    label="Download All Results (CSV)",
                                    data=csv,
                                    file_name=f"batch_query_fanout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    else:
                        st.warning("No valid queries found in the selected column")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with tab3:
    st.markdown("## Understanding Query Fan-Out")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What is Query Fan-Out?
        Query fan-out is how modern AI search systems expand a single user query 
        into multiple related searches to better understand intent and provide relevant and comprehensive results.
        ### Why It Matters for SEO
        Traditional SEO focuses on ranking for specific keywords. But AI search systems:
        - Generate multiple synthetic queries from each user search
        - Pull content from highly relevant content passages for these expanded queries
        - Synthesize answers from multiple sources that best match the queries' **latent semantic meaning**
        **This means:** Your content needs to be relevant for a broader set of related queries at the passage level, not just the user's initial query.
        ### The 6 Query Types Explained
        """)
        for qtype, info in QUERY_TYPE_DESCRIPTIONS.items():
            with st.container():
                st.markdown(f"#### {info['name']}")
                st.markdown(f"**What it is:** {info['description']}")
                st.markdown(f"**Example:** {info['example']}")
                st.markdown(f"**Why it matters:** {info['strategy']}")
                st.markdown("---")
