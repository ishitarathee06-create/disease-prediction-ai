import pandas as pd
import os


def load_history():

    file = "history/history.csv"

    if not os.path.exists(file):

        return pd.DataFrame()

    return pd.read_csv(file)
