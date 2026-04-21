# %%
def detect_passive(srl_output):
    flags = []
    for verb_frame in srl_output["verbs"]:
        tags = verb_frame["tags"]
        description = verb_frame["description"]
        has_arg0 = any("ARG0" in t for t in tags)
        has_arg1 = any("ARG1" in t for t in tags)
        # Passive: victim is grammatical subject, no explicit agent
        if has_arg1 and not has_arg0:
            flags.append({
                "type": "passive_no_agent",
                "verb": verb_frame["verb"],
                "description": description
            })
        # Passive WITH agent (still obfuscatory, but less so)
        elif has_arg1 and has_arg0:
            # Check if verb is in passive form using description markers
            if "[was]" in description or "[were]" in description or "[is]" in description:
                flags.append({
                    "type": "passive_with_agent",
                    "verb": verb_frame["verb"],
                    "description": description
                })
    return flags
# %%
def detect_no_agent(srl_flags, sentence, coref_clusters, police_terms):
    """
    If the sentence has a passive_no_agent flag AND no coref cluster
    in the sentence resolves to a police-related entity, that's a no-agent construction.
    """
    results = []
    for flag in srl_flags:
        if flag["type"] == "passive_no_agent":
            # Check if any coref mention in this sentence refers to police
            police_mentioned = any(
                any(term in mention.lower() for term in police_terms)
                for cluster in coref_clusters
                for mention in cluster
                if mention.lower() in sentence.lower()
            )
            results.append({
                **flag,
                "police_coref_present": police_mentioned,
                "obfuscation_level": "passive_with_agent" if police_mentioned else "no_agent"
            })
    return results
# %%
import re
import spacy
nlp = spacy.load("en_core_web_sm")

NOMINALIZATION_PATTERNS = [
    r"\bofficer[-\s]involved\s+(shooting|killing|incident|death)\b",
    r"\b(fatal|deadly|officer-involved)\s+(shooting|incident|encounter)\b",
    r"\b(use[-\s]of[-\s]force)\s+incident\b",
    r"\b(shooting|killing)\s+death\b",
]

def detect_nominalizations(sentence):
    flags = []
    sentence_lower = sentence.lower()
    for pattern in NOMINALIZATION_PATTERNS:
        matches = re.findall(pattern, sentence_lower)
        if matches:
            flags.append({
                "type": "nominalization",
                "matches": matches,
                "pattern": pattern
            })
    # Also use spacy: look for deverbal nouns (NN tokens with verb roots)
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "NOUN" and token.lemma_ in {"killing", "shooting", "death", "incident"}:
            flags.append({
                "type": "nominalization_spacy",
                "token": token.text,
                "lemma": token.lemma_
            })
    return flags
# %%
INTRANSITIVE_DEATH_VERBS = {"die", "pass", "perish", "succumb", "expire"}
TRANSITIVE_HARM_VERBS = {
    "shoot", "fire", "wound", "injure", "assault", "attack", "pistol-whip", "strike"
}
TRANSITIVE_KILLING_VERBS = {"kill", "shoot", "murder", "slay", "execute"}

def detect_intransitive_verbs(srl_output, sentence):
    flags = []
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            lemma = token.lemma_.lower()
            if lemma in INTRANSITIVE_DEATH_VERBS:
                # Confirm it's used intransitively (no direct object)
                has_obj = any(child.dep_ in ("dobj", "nsubjpass") for child in token.children)
                flags.append({
                    "type": "intransitive_death_verb",
                    "verb": token.text,
                    "lemma": lemma,
                    "has_object": has_obj
                })
            elif lemma in TRANSITIVE_KILLING_VERBS:
                flags.append({
                    "type": "transitive_kill_verb",
                    "verb": token.text,
                    "lemma": lemma
                })
    return flags
# %%
from nltk.tokenize import sent_tokenize
from fastcoref import FCoref
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import spacy
import json
from data import data

coref_model = FCoref(model_name_or_path='/root/.cache/f-coref')
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

TRANSITIVE_KILL_VERBS = {
    "kill", "shoot", "murder", "slay", "execute"
}

TRANSITIVE_HARM_VERBS = {
    "shoot", "fire", "wound", "injure", "assault", "attack", "pistol-whip", "strike"
}

TRANSITIVE_THREAT_VERBS = {
    "brandish", "point", "draw", "display", "aim", "threaten",
    "menace", "confront", "produce", "exhibit", "flash"
}

FOCAL_VERB_LEMMAS = TRANSITIVE_KILL_VERBS | TRANSITIVE_HARM_VERBS | TRANSITIVE_THREAT_VERBS

POLICE_TERMS = {"officer", "police", "cop", "deputy", "agent", "trooper", "detective", "agent"}
VICTIM_TERMS = {"victim", "man", "woman", "person", "suspect", "individual", "deceased"}
WEAPON_TERMS = {"gun", "firearm", "weapon", "pistol", "rifle", "shot", "shots"}

def get_event_entity_spans(clusters, full_text):
    """
    From coref clusters, identify which clusters refer to
    the officer and victim entities in this document.
    Returns the set of mention strings for each.
    """
    officer_mentions = set()
    victim_mentions = set()

    for cluster in clusters:
        cluster_lower = [m.lower() for m in cluster]
        if any(term in mention for term in POLICE_TERMS
               for mention in cluster_lower):
            officer_mentions.update(cluster)
        if any(term in mention for term in VICTIM_TERMS
               for mention in cluster_lower):
            victim_mentions.update(cluster)

    return officer_mentions, victim_mentions

def is_event_relevant(sentence, srl_output, officer_mentions, victim_mentions):
    sentence_lower = sentence.lower()

    officer_in_sentence = any(m.lower() in sentence_lower for m in officer_mentions)
    victim_in_sentence = any(m.lower() in sentence_lower for m in victim_mentions)

    if not (officer_in_sentence or victim_in_sentence):
        return False

    for verb_frame in srl_output["verbs"]:
        # Lemmatize the inflected verb form before checking
        verb_doc = nlp(verb_frame["verb"])
        verb_lemma = verb_doc[0].lemma_.lower()

        if verb_lemma not in FOCAL_VERB_LEMMAS:
            continue
        frame_text = verb_frame["description"].lower()
        if any(m.lower() in frame_text for m in officer_mentions | victim_mentions):
            return True

    return False

results = []

for doc_id, doc in data.items():
    full_text = doc["text"]
    sentences = sent_tokenize(full_text)
    coref_preds = coref_model.predict([full_text])
    clusters = coref_preds[0].get_clusters()

    srl_sentences = []
    officer_mentions, victim_mentions = get_event_entity_spans(clusters, full_text)

    for i, sentence in enumerate(sentences):
        srl_output = srl_predictor.predict(sentence=sentence)

        relevant = is_event_relevant(
            sentence, srl_output, officer_mentions, victim_mentions
        )

        if not relevant:
            srl_sentences.append({
                "uid": f"{doc_id}_s{i}",
                "sentence": sentence,
                "srl": srl_output,
                "event_relevant": False,
                "obfuscation_flags": None
            })
            continue

        passive_flags = detect_passive(srl_output)
        no_agent_flags = detect_no_agent(passive_flags, sentence, clusters, POLICE_TERMS)
        nominalization_flags = detect_nominalizations(sentence)
        intransitive_flags = detect_intransitive_verbs(srl_output, sentence)

        obfuscation_flags = {
            "passive_with_agent": any(f["type"] == "passive_with_agent" for f in passive_flags),
            "passive_no_agent": any(f["obfuscation_level"] == "no_agent" for f in no_agent_flags),
            "nominalization": any(
                f["type"] in ("nominalization", "nominalization_spacy") for f in nominalization_flags),
            "intransitive_verb": any(f["type"] == "intransitive_death_verb" for f in intransitive_flags),
        }

        srl_sentences.append({
            "uid": f"{doc_id}_s{i}",
            "sentence": sentence,
            "srl": srl_output,
            "event_relevant": True,
            "obfuscation_flags": obfuscation_flags,
            "any_obfuscation": any(obfuscation_flags.values()),
            "is_first_sentence": (i == 0)
        })

    results.append({
        "doc_id": doc_id,
        "event": doc["event"],
        "coref_clusters": clusters,
        "sentences": srl_sentences,
        "doc_obfuscation": {
            "any_obfuscation": any(
                s["any_obfuscation"] for s in srl_sentences if s["event_relevant"]
            ),
            "passive_with_agent": any(
                s["obfuscation_flags"]["passive_with_agent"] for s in srl_sentences if s["event_relevant"]
            ),
            "passive_no_agent": any(
                s["obfuscation_flags"]["passive_no_agent"] for s in srl_sentences if s["event_relevant"]
            ),
            "nominalization": any(
                s["obfuscation_flags"]["nominalization"] for s in srl_sentences if s["event_relevant"]
            ),
            "intransitive_verb": any(
                s["obfuscation_flags"]["intransitive_verb"] for s in srl_sentences if s["event_relevant"]
            ),
        }
    })

with open("output_obfuscation.json", "w") as f:
    json.dump(results, f, indent=2)