import sys
import json
import re

def renumber_questions(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    question_pattern = re.compile(r"^!!! question(?:\s*(\d+)\))?\s*(.*)", re.IGNORECASE)
    compteur = 1

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            # Join lines to process the cell as a single string
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            # Replace all question lines in the cell
            def repl(match):
                nonlocal compteur
                texte = match.group(2).strip()
                result = f"!!! question {compteur}) {texte}"
                compteur += 1
                return result

            new_source_text = re.sub(
                r"^!!! question(?:\s*(\d+)\))?\s*(.*)",
                repl,
                source_text,
                flags=re.IGNORECASE | re.MULTILINE
            )

            # Split back to lines with original line endings
            cell['source'] = [line + '\n' for line in new_source_text.splitlines()]

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def clean_numbered_titles(ipynb_path):
            with open(ipynb_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            pattern = re.compile(r"^\s*##\s*\d+\)\s*(.*)", re.IGNORECASE)

            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    new_source = []
                    for line in cell.get('source', []):
                        match = pattern.match(line)
                        if match:
                            titre = match.group(1).strip()
                            new_line = f"## {titre}\n"
                            new_source.append(new_line)
                        else:
                            new_source.append(line)
                    cell['source'] = new_source

            with open(ipynb_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)

def renumber_titre_v2(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    # Compile pattern to match only '##' titles, but not '###' or deeper
    titre_pattern = re.compile(r"^(?<!#)##(?!#)\s*(?:[A-Z]\.)?\s*(.*)", re.IGNORECASE)
    compteur = 0
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            # Join lines to process the cell as a single string
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            def titre_repl(match):
                nonlocal compteur
                texte = match.group(1).strip()
                lettre = alphabet[compteur % len(alphabet)]
                compteur += 1
                return f"## {lettre}. {texte}"

            new_source_text = titre_pattern.sub(
                titre_repl,
                source_text
            )
            cell['source'] = [line + '\n' for line in new_source_text.splitlines()]

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_numerotation.py <notebook.ipynb>")
        sys.exit(1)
    renumber_questions(sys.argv[1])
    #clean_numbered_titles(sys.argv[1])
    renumber_titre_v2(sys.argv[1])
    
    print("Renumbering completed successfully.")


    