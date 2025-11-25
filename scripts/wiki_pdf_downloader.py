import json
import os
import requests
import time
from urllib.parse import quote

def clean_filename(name: str) -> str:
    # Keep letters, digits, spaces, -, _, and dot
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")
    return "".join(c for c in name if c in keep).strip()


def find_title_variations(name: str):
    name = name.strip()
    # Some names are all uppercase, try a few variations
    variations = [name.replace(" ", "_")]
    variations.append(name.capitalize().replace(" ", "_"))
    variations.append(name.title().replace(" ", "_"))
    # Also try lower-case variant
    variations.append(name.lower().replace(" ", "_"))
    # Remove duplicates preserving order
    seen = set()
    out = []
    for v in variations:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def download_wiki_pdfs_from_json(json_path: str, output_dir: str, limit: int = 250, delay: float = 1.0):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            molecules = json.load(f)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return

    success_count = 0
    tried_count = 0

    for mol in molecules:
        if success_count >= limit:
            break

        # mol may be a dict or string
        if isinstance(mol, dict):
            name = mol.get('Name') or mol.get('name') or None
        elif isinstance(mol, str):
            name = mol
        else:
            name = None

        if not name or not isinstance(name, str):
            continue

        # Clean for filepath safe name
        safe_name = clean_filename(name)
        if not safe_name:
            continue

        pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")
        if os.path.exists(pdf_path):
            print(f"PDF for {name} already exists, skipping.")
            continue

        print(f"[{success_count}/{limit}] Processing: {name}...")

        variations = find_title_variations(name)
        downloaded = False

        for title in variations:
            # Quote title for URL safe
            url_title = quote(title, safe='_')
            url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{url_title}"
            tried_count += 1

            try:
                # Set timeout to avoid getting stuck and check content type to ensure it's a PDF
                response = requests.get(url, headers={"User-Agent": "ChemNet-Vision/1.0"}, timeout=15)

                if response.status_code == 200 and response.headers.get('Content-Type', '').lower().startswith('application/pdf'):
                    with open(pdf_path, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    success_count += 1
                    downloaded = True
                    print(f"Downloaded PDF for {name} (found as {title})")
                    break
                elif response.status_code == 404:
                    # try next variation
                    continue
                else:
                    print(f"Failed to download {name} (as {title}) – status: {response.status_code}")
            except Exception as e:
                print(f"Request error for {name} (as {title}): {e}")

        if not downloaded:
            print(f"Could not find Wikipedia PDF for {name}")

        # Be nice to the API
        time.sleep(delay)

    print(f"Done. Successfully downloaded {success_count} PDFs (tried {tried_count} attempts).")

if __name__ == "__main__":
    json_file = os.path.join('data', 'molecules.json')
    pdf_folder = os.path.join('data', 'pdfs')
    # Download the first 250 successful PDF downloads (name from JSON)
    download_wiki_pdfs_from_json(json_file, pdf_folder, limit=121, delay=1.0)
