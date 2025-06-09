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

# Initialize dark mode state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle in sidebar
with st.sidebar:
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

# Dynamic CSS based on dark mode state
def get_css_styles(dark_mode):
    if dark_mode:
        return """
        <style>
            .stApp {
                background-color: #1a1a1a !important;
                color: #e5e5e5 !important;
            }
            .stMarkdown, .stMarkdown p, .stMarkdown div {
                color: #e5e5e5 !important;
            }
            .stSelectbox > div > div {
                background-color: #2d2d2d !important;
                color: #e5e5e5 !important;
            }
            .stTextInput > div > div > input {
                background-color: #2d2d2d !important;
                color: #e5e5e5 !important;
                border-color: #444444 !important;
            }
            .stButton > button {
                background-color: #ff8c42 !important;
                color: #1a1a1a !important;
                border: none !important;
            }
            .stButton > button:hover {
                background-color: #ff7a2e !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #2d2d2d !important;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #2d2d2d !important;
                color: #e5e5e5 !important;
            }
            .stTabs [aria-selected="true"] {
                background-color: #ff8c42 !important;
                color: #1a1a1a !important;
            }
            .stExpander {
                background-color: #2d2d2d !important;
                border-color: #444444 !important;
            }
            .stAlert {
                background-color: #23272e !important;
                color: #e5e5e5 !important;
            }
            .stDownloadButton > button {
                background-color: #2d2d2d !important;
                color: #e5e5e5 !important;
                border: 1px solid #444444 !important;
            }
            .stDownloadButton > button:hover {
                background-color: #3d3d3d !important;
                border-color: #ff8c42 !important;
            }
            html, body, [class*="css"] {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 18px;
                line-height: 1.6;
            }
        </style>
        """
    else:
        return """
        <style>
            .stApp {
                background-color: #ffffff !important;
                color: #2d2d2d !important;
            }
            .stButton > button {
                background-color: #dc6b2f !important;
                color: #ffffff !important;
                border: none !important;
            }
            .stButton > button:hover {
                background-color: #c55a24 !important;
            }
            .stTabs [aria-selected="true"] {
                background-color: #dc6b2f !important;
                color: #ffffff !important;
            }
            .stDownloadButton > button {
                background-color: #fafafa !important;
                color: #2d2d2d !important;
                border: 1px solid #e5e5e5 !important;
            }
            .stDownloadButton > button:hover {
                background-color: #f4f4f4 !important;
                border-color: #dc6b2f !important;
            }
            html, body, [class*="css"] {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 18px;
                line-height: 1.6;
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

def generate_synthetic_queries(query: str) -> Tuple[str, int, pd.DataFrame]:
    try:
        prompt = f"""You are an expert at understanding how Google's AI Mode generates synthetic queries through query fan-out.

Given the user query: "{query}"

First, analyze the complexity, breadth, and ambiguity of this query.
Based on this, decide how many synthetic queries are required for comprehensive coverage (typically between 14 and 30, with more for broad/ambiguous queries, fewer for narrow/specific ones).
State the number in "target_query_count" and generate up to that number.

Include as many queries as possible that are:
- Distinct in their wording or underlying intent (it’s okay if some overlap, as long as they would return different results in search)
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

        # Embedding-based deduplication (looser threshold)
        queries = df['query'].tolist()
        if queries:
            dedup_threshold = 0.85
            model_emb = SentenceTransformer('all-MiniLM-L6-v2')
            emb = model_emb.encode(queries, convert_to_tensor=True)
            keep_indices = []
            for idx, e in enumerate(emb):
                if not keep_indices:
                    keep_indices.append(idx)
                else:
                    sims = util.pytorch_cos_sim(e, emb[keep_indices])
                    if torch.max(sims) < dedup_threshold:
                        keep_indices.append(idx)
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
    st.markdown(f"""
    <div class="generation-stats">
        <p><strong>Attempted to generate:</strong> {target_count} queries</p>
        <p><strong>Successfully generated:</strong> {len(df)} queries</p>
        <p><strong>Generation rate:</strong> {(len(df)/target_count*100):.1f}%</p>
        <p style="font-size: 15px; color: #666; margin-top: 0.5rem;">
        The AI determines the optimal number of queries based on complexity. 
        Actual results may vary to maintain quality over quantity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### AI's Reasoning")
    st.info(reasoning)

    # Distinct card for every query
    st.markdown("### Generated Synthetic Queries")
    for idx, row in df.iterrows():
        st.markdown(f"""
        <div style="
            border: 2px solid var(--border-color);
            border-radius: 10px;
            padding: 1.3rem;
            margin-bottom: 1.5rem;
            background-color: var(--bg-secondary);
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem;">{row['query']}</div>
            <div style="margin-bottom: 0.2rem;">
                <span style="font-weight: 600; color: #aaa;">Type:</span>
                <span style="color: #0066cc; font-weight: 600;">{row['type'].capitalize()}</span>
            </div>
            <div style="margin-bottom: 0.2rem;">
                <span style="font-weight: 600; color: #aaa;">Intent:</span>
                <span>{row['user_intent']}</span>
            </div>
            <div style="margin-bottom: 0.1rem;">
                <span style="font-weight: 600; color: #aaa;">Reasoning:</span>
                <span style="color: var(--text-secondary);">{row['reasoning']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("View all queries as table", expanded=False):
        st.dataframe(df[['query', 'type', 'user_intent', 'reasoning']])

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
                                with st.expander(f"{orig_query} (Generated: {actual}/{target})"):
                                    st.caption(f"Attempted: {target} queries | Generated: {actual} queries | Success rate: {(actual/target*100):.1f}%")
                                    for _, row in query_results.iterrows():
                                        st.markdown(f"""
                                        <div style="
                                            border: 2px solid var(--border-color);
                                            border-radius: 10px;
                                            padding: 1.3rem;
                                            margin-bottom: 1.5rem;
                                            background-color: var(--bg-secondary);
                                            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                                        ">
                                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--accent-color); margin-bottom: 0.25rem;">{row['query']}</div>
                                            <div style="margin-bottom: 0.2rem;">
                                                <span style="font-weight: 600; color: #aaa;">Type:</span>
                                                <span style="color: #0066cc; font-weight: 600;">{row['type'].capitalize()}</span>
                                            </div>
                                            <div style="margin-bottom: 0.2rem;">
                                                <span style="font-weight: 600; color: #aaa;">Intent:</span>
                                                <span>{row['user_intent']}</span>
                                            </div>
                                            <div style="margin-bottom: 0.1rem;">
                                                <span style="font-weight: 600; color: #aaa;">Reasoning:</span>
                                                <span style="color: var(--text-secondary);">{row['reasoning']}</span>
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
