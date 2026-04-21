from nltk.tokenize import sent_tokenize
from fastcoref import FCoref
import json
from data import data

coref_model = FCoref()

results = []

for doc_id, doc in data.items():
    full_text = doc["text"]
    sentences = sent_tokenize(full_text)

    coref_preds = coref_model.predict([full_text])
    clusters = coref_preds[0].get_clusters()

    srl_sentences = []
    for i, sentence in enumerate(sentences):
        srl_output = srl_predictor.predict(sentence=sentence)
        srl_sentences.append({
            "uid": f"{doc_id}_s{i}",
            "sentence": sentence,
            "srl": srl_output
        })

    results.append({
        "doc_id": doc_id,
        "event": doc["event"],
        "coref_clusters": clusters,
        "sentences": srl_sentences
    })

with open("output2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. Processed {len(results)} documents.")