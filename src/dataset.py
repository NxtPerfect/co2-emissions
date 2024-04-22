import numpy as np
import pandas as pd

def loadCSV(path: str):
    data = pd.read_csv(path)
    print(data)

loadCSV("data/emissions.csv")
