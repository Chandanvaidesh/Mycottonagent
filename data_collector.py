import json
from pathlib import Path

from transformers import AutoTokenizer
import os

chunks_dir = Path("pest_chunks")
os.makedirs(chunks_dir, exist_ok=True)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512
OVERLAP=50

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID, use_fast=True)

def chunk_text_by_tokens(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    if not text.strip():
        return
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,  # don't cut off
        return_attention_mask=False,
        return_token_type_ids=False
    )
    tokens = encoding["input_ids"]  # token IDs without warning
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
        if start >= len(tokens):
            break
    return chunks





def extract_text_from_docling_json(obj):
    texts = []
    if isinstance(obj,dict):
        for key, value in obj.items():
            if key.lower() in ["text","content"] and isinstance(value,str):
                texts.append(value.strip())
            else:
                texts.extend(extract_text_from_docling_json(value))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_from_docling_json(item))
    return texts


json_path = Path("Cleaned_pest")
json_folder = list(json_path.rglob("*.json"))


for file in json_folder:
    try:
        with open(file,"r",encoding="utf-8") as f:
            data = json.load(f)
        all_texts = extract_text_from_docling_json(data)
        text_data = "\n".join(all_texts).strip()

        
        chunks= chunk_text_by_tokens(text_data)
        output_file = chunks_dir/f"{file.stem}_chunks.json"
        with open(output_file,"w", encoding="utf-8") as out_f:
            json.dump(chunks,out_f, ensure_ascii=False,indent=2)


    except Exception as e:
        print(f"Failed to process{file.name}:{e}")

