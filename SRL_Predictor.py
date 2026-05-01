import pandas as pd
from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import json

srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

df = pd.read_csv("SRL_input_extracted.csv")

results = []

for _, row in df.iterrows():
    sentences = sent_tokenize(row["text"])

    srl_sentences = []
    for i, sentence in enumerate(sentences):
        srl_output = srl_predictor.predict(sentence=sentence)
        srl_sentences.append({
            "uid": f"{row['id']}_s{i}",
            "sentence": sentence,
            "srl": srl_output,
        })

        print(f"\n--- {row['id']}_s{i} ---")
        print(f"SENTENCE: {sentence}")
        for verb_frame in srl_output["verbs"]:
            print(f"  [{verb_frame['verb']}] {verb_frame['description']}")

    results.append({
        "doc_id": row["id"],
        "severity": row["severity"],
        "victim": row["victim"],
        "sentences": srl_sentences,
    })

with open("output_srl.json", "w") as f:
    json.dump(results, f, indent=2)