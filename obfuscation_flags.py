import json
import re
import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")

PASSIVE_VERBS = {"kill", "gun", "murder", "shoot", "hit", "fire", "strike", "discharge"}
INTRANSITIVE_VERBS = {"die"}
DECLARED_DEAD_VERBS = {"declare", "find", "pronounce"}
KNOWN_AGENTS = {"law_enforcement_agent", "law_enforcement_agents", "they", "swat"}

DEBUG_FILE = open("debug_passive.txt", "w")


def is_passive(verb_frame, sentence):
    description = verb_frame["description"].lower()
    verb_doc = nlp(verb_frame["verb"])
    verb_lemma = verb_doc[0].lemma_.lower()

    in_passive_verbs = verb_lemma in PASSIVE_VERBS
    pattern = r'\b(was|were|is|are|be)\s+\[v:'
    regex_match = bool(re.search(pattern, description))

    DEBUG_FILE.write(f"  checking: {verb_frame['verb']} | lemma: {verb_lemma} | in PASSIVE_VERBS: {in_passive_verbs}\n")
    DEBUG_FILE.write(f"  description: {description}\n")
    DEBUG_FILE.write(f"  regex match: {regex_match}\n\n")

    if verb_lemma not in PASSIVE_VERBS and "open fire" not in sentence.lower():
        return False
    return bool(re.search(pattern, description))




# def is_passive(verb_frame, sentence):
#     description = verb_frame["description"].lower()
#     verb_doc = nlp(verb_frame["verb"])
#     verb_lemma = verb_doc[0].lemma_.lower()
#
#     if verb_lemma not in PASSIVE_VERBS and "open fire" not in sentence.lower():
#         return False
#
#     # check for "was/were/is/are/be" immediately before [v: in the description
#     return bool(re.search(r'\b(was|were|is|are|be)\s+\[v:', description))

# def detect_passive(srl_output, sentence):
#     """
#     Passive: passive voice used with a transitive verb from PASSIVE_VERBS
#     and ARG1 (patient) contains VICTIM.
#     """
#     flags = []
#     for verb_frame in srl_output["verbs"]:
#         if not is_passive(verb_frame, sentence):
#             continue
#         description = verb_frame["description"].lower()
#         tags = verb_frame["tags"]
#         # check patient (ARG1) contains VICTIM
#         arg1_tokens = [
#             srl_output["words"][i]
#             for i, tag in enumerate(tags)
#             if "ARG1" in tag
#         ]
#         arg1_text = " ".join(arg1_tokens).lower()
#         if "victim" not in arg1_text:
#             continue
#         # extract agent (ARG0) if present
#         arg0_tokens = [
#             srl_output["words"][i]
#             for i, tag in enumerate(tags)
#             if "ARG0" in tag
#         ]
#         arg0_text = " ".join(arg0_tokens).lower()
#         flags.append({
#             "type": "passive",
#             "verb": verb_frame["verb"],
#             "arg0": arg0_text,
#             "arg1": arg1_text,
#             "description": verb_frame["description"]
#         })
#     return flags

def detect_passive(srl_output, sentence):
    """
    Passive: passive voice used with a transitive verb from PASSIVE_VERBS.
    Excludes cases where ARG1 is LAW_ENFORCEMENT_AGENT/AGENTS.
    """
    flags = []
    for verb_frame in srl_output["verbs"]:
        if not is_passive(verb_frame, sentence):
            continue
        print(f"  PASSED is_passive: {verb_frame['verb']}")  # add this
        tags = verb_frame["tags"]
        arg1_tokens = [srl_output["words"][i] for i, tag in enumerate(tags) if "ARG1" in tag]
        arg0_tokens = [srl_output["words"][i] for i, tag in enumerate(tags) if "ARG0" in tag]
        arg1_text = " ".join(arg1_tokens).lower()
        arg0_text = " ".join(arg0_tokens).lower()
        if "law_enforcement_agent" in arg1_text:
            continue
        if "nobody" or "no one" in arg1_text:
            continue
        flags.append({
            "type": "passive",
            "verb": verb_frame["verb"],
            "arg0": arg0_text,
            "arg1": arg1_text,
            "description": verb_frame["description"]
        })
    return flags

def detect_no_agent(passive_flags):
    """
    No agent: subset of passive sentences where ARG0 is absent
    or not a known agent (LAW_ENFORCEMENT_AGENT, they, SWAT etc).
    """
    flags = []
    for flag in passive_flags:
        arg0 = flag["arg0"].strip()
        if not arg0 or not any(agent in arg0 for agent in KNOWN_AGENTS):
            flags.append({
                **flag,
                "type": "no_agent",
            })
    return flags

def detect_intransitive(srl_output, sentence):
    """
    Intransitive: die/dead used with VICTIM as subject,
    or (declared/found/pronounced) dead with VICTIM as patient.
    """
    flags = []
    sentence_lower = sentence.lower()

    for verb_frame in srl_output["verbs"]:
        verb_doc = nlp(verb_frame["verb"])
        verb_lemma = verb_doc[0].lemma_.lower()
        tags = verb_frame["tags"]

        # (declared/found/pronounced) dead — VICTIM in ARG1, "dead" in ARG2
        if verb_lemma in DECLARED_DEAD_VERBS:
            arg1_tokens = [srl_output["words"][i] for i, tag in enumerate(tags) if "ARG1" in tag]
            arg2_tokens = [srl_output["words"][i] for i, tag in enumerate(tags) if "ARG2" in tag]
            arg1_text = " ".join(arg1_tokens).lower()
            arg2_text = " ".join(arg2_tokens).lower()
            if "victim" in arg1_text and "dead" in arg2_text:
                flags.append({
                    "type": "intransitive",
                    "pattern": "declared/found/pronounced dead",
                    "arg1": arg1_text,
                    "description": verb_frame["description"]
                })

        # die — VICTIM in ARG1
        elif verb_lemma in INTRANSITIVE_VERBS:
            arg1_tokens = [srl_output["words"][i] for i, tag in enumerate(tags) if "ARG1" in tag or "ARG0" in tag]
            arg1_text = " ".join(arg1_tokens).lower()
            if "victim" in arg1_text:
                flags.append({
                    "type": "intransitive",
                    "verb": verb_frame["verb"],
                    "arg": arg1_text,
                    "description": verb_frame["description"]
                })

    # "is dead" — loose check since copular constructions don't SRL cleanly
    if re.search(r"\bis\s+dead\b", sentence_lower) and "victim" in sentence_lower:
        flags.append({
            "type": "intransitive",
            "pattern": "is dead",
            "sentence": sentence
        })

    return flags

def detect_nominalization(sentence):
    """
    Nominalization: [X]-involved shooting, [X]-related shooting,
    shooting (death/killing) of.
    """
    flags = []
    sentence_lower = sentence.lower()
    patterns = [
        r"\S+-involved\s+shooting\b",
        r"\S+-related\s+shooting\b",
        r"\bshooting\s+(death\s+)?of\b",
        r"\bshooting\s+(killing\s+)?of\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence_lower)
        if match:
            flags.append({
                "type": "nominalization",
                "match": match.group(),
                "pattern": pattern
            })
    return flags

def flag_obfuscation(sentence, srl_output):
    passive_flags = detect_passive(srl_output, sentence)
    no_agent_flags = detect_no_agent(passive_flags)
    intransitive_flags = detect_intransitive(srl_output, sentence)
    nominalization_flags = detect_nominalization(sentence)

    return {
        "passive": passive_flags,
        "no_agent": no_agent_flags,
        "intransitive": intransitive_flags,
        "nominalization": nominalization_flags,
        "any_obfuscation": any([
            passive_flags,
            no_agent_flags,
            intransitive_flags,
            nominalization_flags
        ])
    }


nlp = spacy.load("en_core_web_sm")

with open("output_srl.json", "r") as f:
    results = json.load(f)

for doc in results:
    for sentence_data in doc["sentences"]:
        sentence = sentence_data["sentence"]
        srl_output = sentence_data["srl"]
        obfuscation = flag_obfuscation(sentence, srl_output)
        sentence_data["obfuscation"] = obfuscation

        print(f"\n--- {sentence_data['uid']} ---")
        print(f"SENTENCE: {sentence}")
        for verb_frame in srl_output["verbs"]:
            print(f"  [{verb_frame['verb']}] {verb_frame['description']}")
        if obfuscation["any_obfuscation"]:
            print(f"  ⚑ OBFUSCATION: { {k: v for k, v in obfuscation.items() if k != 'any_obfuscation' and v} }")

# at the end of your script
DEBUG_FILE.close()

with open("output_obfuscation.json", "w") as f:
    json.dump(results, f, indent=2)