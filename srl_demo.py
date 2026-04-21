
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from nltk.tokenize import sent_tokenize
import json
from data import data

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

results = []

for doc_id, doc in data.items():
    sentences = sent_tokenize(doc["text"])
    for i, sentence in enumerate(sentences):
        srl_output = predictor.predict(sentence=sentence)
        results.append({
            "uid": f"{doc_id}_s{i}",
            "event": doc["event"],
            "sentence": sentence,
            "srl": srl_output
        })

# Save to JSON
with open("output.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. Processed {len(results)} sentences.")