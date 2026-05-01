import pandas as pd
import re

def extract_story(text):
    # find first and last occurrence of either token
    indices = [m.start() for m in re.finditer(r'VICTIM|LAW_ENFORCEMENT', text)]

    if not indices:
        return None

    first = indices[0]
    last = indices[-1]

    # walk back to start of sentence containing first match
    start = text.rfind(".", 0, first)
    start = 0 if start == -1 else start + 2

    # walk forward to end of sentence containing last match
    end = text.find(".", last)
    end = len(text) if end == -1 else end + 1

    return text[start:end]

df = pd.read_csv("SRL_input.csv")

rows = []
for _, row in df.iterrows():
    trimmed = extract_story(row["text"])
    if trimmed is None:
        print(f"Warning: no VICTIM or LAW_ENFORCEMENT found in doc {row['id']}, skipping")
        continue
    rows.append({
        "id": row["id"],
        "story": trimmed,
        "text": row["text"],
        "severity": row["severity"],
        "victim": row["victim"]
    })

df_extracted = pd.DataFrame(rows)
df_extracted.to_csv("SRL_input_extracted.csv", index=False)
print(f"Saved {len(df_extracted)} rows to SRL_input_extracted.csv")