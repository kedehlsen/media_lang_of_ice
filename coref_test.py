from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor
from fastcoref import FCoref
import pandas as pd
import json
import re

# --------------------
# Load models
# --------------------
# srl_predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
# )

coref_model = FCoref(model_name_or_path="/root/.cache/f-coref")

# --------------------
df = pd.read_csv("data_set.csv")
df_filtered = df[df["severity"].isin({"shooting", "fatal", "shooting (injury)"})]
data_row = df_filtered.iloc[0]

full_text = data_row["text"]

print(full_text)

# coref_preds = coref_model.predict([full_text])
# coref_indices = coref_preds[0].get_clusters(as_strings=False)

import coreferee, spacy
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
doc = nlp(full_text)
doc._.coref_chains.print()

print(coref_indices)

print(coref_preds)