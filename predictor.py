from utils.history_logger import save_history
import pandas as pd
import joblib
import numpy as np

from nlpe import SymptomExtractor


class DiseasePredictor:

    def __init__(self):

        # Load trained model
        self.model = joblib.load("best_model.pkl")
        self.mlb = joblib.load("symptom_encoder.pkl")
        self.le = joblib.load("label_encoder.pkl")

        # Load datasets
        self.description = pd.read_csv("dataset/symptom_Description.csv")
        self.precautions = pd.read_csv("dataset/symptom_precaution.csv")

        # NLP extractor
        self.extractor = SymptomExtractor(self.mlb.classes_)

    def predict(self, text):

        # -------------------------
        # Extract Symptoms
        # -------------------------
        symptoms = self.extractor.extract(text)

        if len(symptoms) == 0:

            return {
                "disease": "No symptoms detected",
                "description": "Please enter valid symptoms.",
                "precautions": [],
                "confidence": 0,
                "symptoms": [],
                "top3": []
            }

        if len(symptoms) < 3:

            return {
                "disease": "Insufficient symptoms",
                "description": "Please enter at least 3 symptoms.",
                "precautions": [],
                "confidence": 0,
                "symptoms": symptoms,
                "top3": []
            }

        # -------------------------
        # Convert symptoms
        # -------------------------
        X = self.mlb.transform([symptoms])

        # -------------------------
        # Prediction probabilities
        # -------------------------
        probabilities = self.model.predict_proba(X)[0]

        # Top 3 predictions
        top3_indices = np.argsort(probabilities)[::-1][:3]

        top3_predictions = []

        for idx in top3_indices:

            disease_name = self.le.inverse_transform([idx])[0]

            confidence = round(probabilities[idx] * 100, 2)

            top3_predictions.append({
                "disease": disease_name,
                "confidence": confidence
            })

        # Best prediction
        disease = top3_predictions[0]["disease"]
        confidence = top3_predictions[0]["confidence"]

        # -------------------------
        # Disease description
        # -------------------------
        desc = self.description[
            self.description["Disease"] == disease
        ]

        if not desc.empty:
            description = desc["Description"].values[0]
        else:
            description = "Description not available."

        # -------------------------
        # Precautions
        # -------------------------
        pre = self.precautions[
            self.precautions["Disease"] == disease
        ]

        precaution_list = []

        if not pre.empty:

            for i in range(1, 5):

                col = f"Precaution_{i}"

                if col in pre.columns:

                    value = pre[col].values[0]

                    if pd.notna(value):
                        precaution_list.append(str(value))
        save_history(
            text,
            symptoms,
            disease,
            confidence
        )

        # -------------------------
        # Return everything
        # -------------------------
        return {
            "disease": disease,
            "description": description,
            "precautions": precaution_list,
            "confidence": confidence,
            "symptoms": symptoms,
            "top3": top3_predictions
        }