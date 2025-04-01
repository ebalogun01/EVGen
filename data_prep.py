import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

df_raw = pd.read_csv("Data/raw_google_data.csv")
df_clean = pd.read_csv("Data/clean_google_data.csv")

data = pd.read_csv("Data/chargeDataCleanedWithSessionID.csv")
with open("Data/chargeDriverZipData.json", "r") as read_file:
    driver_ids = json.load(read_file)

data_npy = data.to_numpy().T[0:18944, :]
np.save("paper_imgs/chargeDataCleanedWithSessionID.npy", data_npy)




