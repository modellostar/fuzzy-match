import json

desired_types = {"T047", "T048", "T046", "T191"}

input_file = r"attached_assets\umls_2022_ab_cat0129.txt"       # Replace with the path to your input file
output_file = "output.txt"     # Replace with desired path for the output file

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue  # Skip empty or blank lines
        
        # Parse the line as JSON
        data = json.loads(line)
        
        # Check if there's an overlap between the line's 'types' and the desired types
        if any(t in desired_types for t in data.get("types", [])):
            # Write the matched line as JSON to output_file
            outfile.write(json.dumps(data, ensure_ascii=False))
            outfile.write("\n")
