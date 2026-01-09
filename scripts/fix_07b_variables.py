"""
Fix all variable definition issues in Notebook 07b
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "07b_maml_xuetangx_evaluation.ipynb"

print(f"Fixing: {NB_PATH}")

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix CELL 07-07b
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell['source'] and 'CELL 07-07b' in cell['source'][0]:
        print(f"\nFixing CELL 07-07b at index {i}")

        # Find the line with "# Extract model dimensions for functional_forward"
        new_source = []
        found_extract_section = False

        for line in cell['source']:
            new_source.append(line)

            # Add proper variable definitions after criterion
            if 'criterion = nn.CrossEntropyLoss()' in line and not found_extract_section:
                new_source.append('\n')
                new_source.append('# Extract model dimensions for functional_forward (n_items already defined in CELL 07-04)\n')
                new_source.append('hidden_dim = CFG["gru_config"]["hidden_dim"]\n')
                new_source.append('\n')
                found_extract_section = True

            # Remove old incorrect definitions
            if 'vocab_course_id_to_idx' in line:
                new_source.pop()  # Remove this line

        cell['source'] = new_source
        print("+ Added hidden_dim definition")
        print("+ Removed incorrect vocab line")
        print("+ n_items is already defined in CELL 07-04")
        break

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[SUCCESS] Fixed {NB_PATH}")
print("\nVariables now properly defined:")
print("  - n_items: defined in CELL 07-04 (len(course2id))")
print("  - hidden_dim: defined in CELL 07-07b (from config)")
print("  - inner_lr: defined in CELL 07-07b (from config)")
print("  - num_inner_steps: defined in CELL 07-07b (from config)")
print("  - max_seq_len: defined in CELL 07-07b (from config)")
print("  - criterion: defined in CELL 07-07b")
