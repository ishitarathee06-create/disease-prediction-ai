import pandas as pd
from datetime import datetime
import os


def save_history(user_input, symptoms, disease, confidence):

    file_path = "history/history.csv"

    row = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Input": user_input,
        "Detected Symptoms": ", ".join(symptoms),
        "Disease": disease,
        "Confidence": confidence
    }

    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

    else:

        df = pd.DataFrame(columns=[
            "Date",
            "Input",
            "Detected Symptoms",
            "Disease",
            "Confidence"
        ])

    df.loc[len(df)] = row

    df.to_csv(file_path, index=False)
