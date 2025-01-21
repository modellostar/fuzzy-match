import pandas as pd
import json

# Initialize a list to hold rows for our DataFrame
rows = []

# Read the JSONL file line by line
with open("attached_assets\output.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        cui = data["concept_id"]
        
        # Merge canonical_name into the aliases list
        # (We make it a set first, to avoid duplicates if the canonical_name is already in aliases)
        aliases = set(data.get("aliases", []))
        aliases.add(data["canonical_name"])
        
        # Create a row for each alias
        for alias in aliases:
            rows.append({"cui": cui, "alias": alias})

# Convert to DataFrame
df = pd.DataFrame(rows, columns=["cui", "alias"])

# Show the result
df.to_csv('cui-alias-mapping.csv',encoding="utf-8")
