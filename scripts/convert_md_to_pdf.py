#!/usr/bin/env python3
"""
Convert Markdown justification document to PDF.
Requires: pip install markdown2 weasyprint
"""

import markdown2
from pathlib import Path
from weasyprint import HTML, CSS

# Paths
REPO_ROOT = Path(__file__).parent.parent
MD_PATH = REPO_ROOT / "docs" / "meta_learning_architecture_justification.md"
PDF_PATH = REPO_ROOT / "docs" / "meta_learning_architecture_justification.pdf"

# Read markdown
with MD_PATH.open("r", encoding="utf-8") as f:
    md_content = f.read()

# Convert to HTML
html_content = markdown2.markdown(
    md_content,
    extras=["tables", "fenced-code-blocks", "header-ids", "toc"]
)

# Add CSS styling for professional look
css_style = """
@page {
    size: A4;
    margin: 2cm;
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

h1 {
    color: #1a1a1a;
    font-size: 24pt;
    border-bottom: 3px solid #2c5aa0;
    padding-bottom: 10px;
    margin-top: 30px;
    page-break-before: auto;
}

h2 {
    color: #2c5aa0;
    font-size: 18pt;
    margin-top: 25px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
}

h3 {
    color: #4a4a4a;
    font-size: 14pt;
    margin-top: 20px;
}

h4 {
    color: #666;
    font-size: 12pt;
    margin-top: 15px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 10pt;
}

th {
    background-color: #2c5aa0;
    color: white;
    padding: 10px;
    text-align: left;
    font-weight: bold;
}

td {
    border: 1px solid #ddd;
    padding: 8px;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
}

pre {
    background-color: #f4f4f4;
    padding: 15px;
    border-left: 4px solid #2c5aa0;
    overflow-x: auto;
    font-size: 9pt;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 4px solid #ccc;
    margin-left: 0;
    padding-left: 20px;
    color: #666;
    font-style: italic;
}

.checkmark {
    color: #28a745;
    font-weight: bold;
}

.crossmark {
    color: #dc3545;
    font-weight: bold;
}

.warning {
    color: #ffc107;
    font-weight: bold;
}

hr {
    border: none;
    border-top: 2px solid #ccc;
    margin: 30px 0;
}

a {
    color: #2c5aa0;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
"""

# Wrap HTML with proper structure
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Meta-Learning Architecture Design Justification</title>
    <style>{css_style}</style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Convert to PDF
print(f"Converting {MD_PATH} to PDF...")
HTML(string=full_html).write_pdf(PDF_PATH)

print(f"✓ PDF created: {PDF_PATH}")
print(f"  Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")