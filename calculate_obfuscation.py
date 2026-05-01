import json
import pandas as pd

with open("output_obfuscation.json", "r") as f:
    results = json.load(f)

rows = []

for doc in results:
    sentences = doc["sentences"]
    total = len(sentences)

    passive = sum(1 for s in sentences if s.get("obfuscation") and s["obfuscation"]["passive"])
    no_agent = sum(1 for s in sentences if s.get("obfuscation") and s["obfuscation"]["no_agent"])
    intransitive = sum(1 for s in sentences if s.get("obfuscation") and s["obfuscation"]["intransitive"])
    nominalization = sum(1 for s in sentences if s.get("obfuscation") and s["obfuscation"]["nominalization"])
    any_obfuscation = sum(1 for s in sentences if s.get("obfuscation") and s["obfuscation"]["any_obfuscation"])

    rows.append({
        "doc_id": doc["doc_id"],
        "total_sentences": total,
        "passive": passive,
        "no_agent": no_agent,
        "intransitive": intransitive,
        "nominalization": nominalization,
        "any_obfuscation": any_obfuscation,
    })

df = pd.DataFrame(rows)
df.to_csv("obfuscation_summary.csv", index=False)
print(df.to_string())