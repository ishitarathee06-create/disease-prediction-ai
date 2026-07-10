import re
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class SymptomExtractor:

    def __init__(self, symptoms):

        # Clean symptom list
        self.symptoms = sorted(
            list(set([s.strip() for s in symptoms if s.strip() != ""]))
        )

        # -------------------------------
        # Phrase Matcher
        # -------------------------------
        self.matcher = PhraseMatcher(
            nlp.vocab,
            attr="LOWER"
        )

        for symptom in self.symptoms:
            phrase = symptom.replace("_", " ")
            self.matcher.add(
                symptom,
                [nlp.make_doc(phrase)]
            )

        # -------------------------------
        # Synonyms Dictionary
        # -------------------------------
        self.synonyms = {

            # Fever
            "fever": "high_fever",
            "temperature": "high_fever",
            "high temperature": "high_fever",
            "high fever": "high_fever",

            # Head
            "head ache": "headache",
            "head pain": "headache",
            "migraine": "headache",

            # Vomiting
            "throwing up": "vomiting",
            "throw up": "vomiting",
            "vomit": "vomiting",

            # Nausea
            "feeling sick": "nausea",
            "queasy": "nausea",

            # Stomach
            "stomach ache": "stomach_pain",
            "belly pain": "stomach_pain",
            "tummy pain": "stomach_pain",
            "abdomen pain": "abdominal_pain",

            # Skin
            "itchy": "itching",
            "itchy skin": "itching",
            "itchiness": "itching",
            "rash": "skin_rash",
            "rashes": "skin_rash",

            # Cold
            "cold": "chills",
            "shivering": "chills",

            # Weakness
            "tired": "fatigue",
            "weak": "fatigue",
            "exhausted": "fatigue",

            # Breathing
            "breathlessness": "breathlessness",
            "difficulty breathing": "breathlessness",
            "shortness of breath": "breathlessness",

            # Cough
            "dry cough": "cough",
            "coughing": "cough"
        }

    def extract(self, text):

        text = text.lower()

        detected = set()

        # --------------------------------
        # Replace synonyms
        # --------------------------------
        for phrase, symptom in self.synonyms.items():
            if phrase in text:
                detected.add(symptom)
                text = text.replace(
                    phrase,
                    symptom.replace("_", " ")
                )

        # --------------------------------
        # Phrase Matching
        # --------------------------------
        doc = nlp(text)

        matches = self.matcher(doc)

        for match_id, start, end in matches:

            symptom = nlp.vocab.strings[match_id]

            detected.add(symptom)

        # --------------------------------
        # Fuzzy Matching (misspellings)
        # --------------------------------
        words = re.findall(r"[a-zA-Z_]+", text)

        for word in words:

            match = process.extractOne(
                word,
                self.symptoms,
                scorer=fuzz.ratio
            )

            if match and match[1] >= 90:
                detected.add(match[0])

        return sorted(list(detected))
