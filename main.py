import streamlit as st
import pandas as pd
from search_engine import SearchEngine
from utils import load_jsonl_data, highlight_match
import json
from typing import Iterator
import io
import requests
import time

# Page config
st.set_page_config(
    page_title="UMLS Concept Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for progress tracking
if 'indexing_progress' not in st.session_state:
    st.session_state.indexing_progress = 0
    st.session_state.indexing_status = ""
    st.session_state.file_processing_progress = 0

# Initialize search engine
if 'search_engine' not in st.session_state:
    # Load default data
    default_data = load_jsonl_data("attached_assets/output.jsonl")
    #st.session_state.search_engine = SearchEngine(default_data)

    st.session_state.search_engine = SearchEngine(default_data, index_path='index\sapbert_index.faiss',concept_alias_mapping_file='attached_assets\output.jsonl')
    st.session_state.current_data = default_data

# Custom CSS
st.markdown("""
    <style>
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #ffd54f;
        padding: 0 2px;
        border-radius: 2px;
    }
    .upload-text {
        color: #666;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload and search settings
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload JSONL file", type=['jsonl'])
    st.markdown('<p class="upload-text">Files larger than 200MB are supported through chunked upload.</p>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            # Create progress tracking elements
            upload_progress = st.progress(0, "Upload Progress")
            index_progress = st.progress(0, "Indexing Progress")
            status_text = st.empty()

            # Start file upload to FastAPI backend
            with st.spinner("Uploading and processing file..."):
                # Upload file
                files = {'file': uploaded_file}
                response = requests.post('http://localhost:8000/upload', files=files)

                if response.status_code == 200:
                    # Monitor processing status
                    while True:
                        status_response = requests.get('http://localhost:8000/status')
                        status = status_response.json()

                        # Update progress bars
                        if status['progress'] < 1.0:
                            upload_progress.progress(status['progress'], text=status['status'])
                            time.sleep(0.1)
                        else:
                            upload_progress.progress(1.0, text="Complete!")
                            break

                    status_text.success(f"✅ Successfully processed file")
                else:
                    error_msg = response.json().get('error', 'Unknown error occurred')
                    status_text.error(f"Error: {error_msg}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.header("Search Settings")
    fuzzy_threshold = st.slider(
        "Fuzzy match threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    semantic_threshold = st.slider(
        "Semantic match threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    max_results = st.number_input(
        "Maximum results",
        min_value=1,
        max_value=50,
        value=10
    )

# Main content area
st.title("UMLS Concept Search")
st.markdown("Search through UMLS concepts using fuzzy and semantic matching")

# Display current dataset info with more detail
total_concepts = len(st.session_state.current_data)
st.info(f"📊 Currently loaded and indexed: {total_concepts:,} concepts")

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter search term", key="search_query")
with col2:
    search_type = st.selectbox(
        "Search type",
        ["Hybrid", "Fuzzy", "Semantic"],
        key="search_type"
    )

# Search results display
if query:
    results = st.session_state.search_engine.search(
        query,
        search_type.lower(),
        fuzzy_threshold,
        semantic_threshold,
        max_results
    )

    if results:
        st.write(f"Found {len(results)} results:")
        for result in results:
            with st.container():
                st.markdown(f"""
                    <div class="result-box">
                        <h3>{highlight_match(result['canonical_name'], query)}</h3>
                        <p><strong>CUI:</strong> {result['concept_id']}</p>
                        <p><strong>Score:</strong> {result['score']:.3f}</p>
                        <p><strong>Match Type:</strong> {result['match_type']}</p>
                        <p><strong>Aliases:</strong> {', '.join(result['aliases'])}</p>
                        {"<p><strong>Definition:</strong> " + result.get('definition', '') + "</p>" if 'definition' in result else ""}
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No results found.")
else:
    st.info("Enter a search term to begin")