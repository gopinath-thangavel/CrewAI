import json

# Load the uploaded notebook
input_path = "/mnt/data/FT_2.ipynb"
output_path = "/mnt/data/FT_2_fixed.ipynb"

with open(input_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Remove metadata.widgets if exists
if "metadata" in notebook and "widgets" in notebook["metadata"]:
    del notebook["metadata"]["widgets"]

# Save the cleaned notebook
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

output_path
