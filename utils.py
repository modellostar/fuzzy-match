import json
from typing import List, Dict, Any
import re

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSONL data from file."""
    data = []
    with open(file_path, 'r',encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

def highlight_match(text: str, query: str) -> str:
    """Highlight matching portions of text."""
    if not query:
        return text
    
    # Escape special regex characters in query
    query_escaped = re.escape(query)
    
    # Create pattern for case-insensitive matching
    pattern = re.compile(f'({query_escaped})', re.IGNORECASE)
    
    # Replace matches with highlighted version
    return pattern.sub(r'<span class="highlight">\1</span>', text)
