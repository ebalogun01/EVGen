"""This file reads raw data from Google EV charging facilities, cleans and parses relevant info. All code here
was used in cleaning and parsing process."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


data1 = pd.read_csv('DataSet/chargeDataCleaned.csv')
data2 = pd.read_csv('MTV-46-2019 Session-Details-Meter-with-Summary_raw.csv')
# data2 = data2['User Id', 'Plug In Event Id', 'Driver Postal Code']
# chargingInfoJSON = {}
# for index, row in data.iterrows():
#     chargingInfoJSON[row['Plug In Event Id']] = [row['Driver Postal Code'], row['User Id']]
#
# json.dump(chargingInfoJSON)
# stop = 0
# for column in data:
#     print(column)
#     if stop == 50:
#         break
#     plt.plot(data[column] * 4)
#     stop += 1
#     plt.xlabel('Time of Day')
#     plt.ylabel('Power (kW)')
# plt.show()

chargeID = data2['Plug In Event Id'].unique()
chargeID.shape = (chargeID.size, )
data1.columns = chargeID


# # # def convertTime()
# count = 0
# dataMatrix = np.zeros((96, 1))
# for idx in chargeID:
#     chargeVector = np.zeros((96,))
#     chargeData = data.loc[data['Plug In Event Id'] == idx]
#     chargeData = chargeData['Energy Consumed (AC kWh)'].to_numpy()
#     chargeData.shape = (chargeData.size,)
#     dataLength = len(chargeData)
#     chargeIdx = (indices[count:count+dataLength])
#     chargeVector[chargeIdx[:]] = chargeData
#     chargeVector.shape = (96, 1)
#     if count == 0:
#         dataMatrix = chargeVector
#     else:
#         dataMatrix = np.hstack([dataMatrix, chargeVector])
#     count += dataLength
#
# np.savetxt("chargeDataCleaned.csv", dataMatrix, delimiter=",")

N = 10 # say we have 10 examples
noise_length = 64
noise_list = []
X = np.random.rand((N, noise_length)) # this
difference_array = np.zeros((N**2, noise_length))
difference_norm_array = np.zeros((N**2, 1))
C_Gap_Array = np.zeros(1, N**2)
C = np.random.randn(1, N)
for i in range(N):
    c = C[1, i]
    c_gap = np.abs(C - c)
    C_Gap_Array[1, i*N:i*N + N] = c_gap
    noise = X[i, :]
    gap = X - noise
    gap_norm = np.linalg.norm(gap, axis=1)  # row-wise L2 norm
    difference_norm_array[i*N:i*N + N, 1] = gap_norm  # can squash this line and the line above into one line of code
    difference_array[i*N:i*N + N, :] = gap # this creates the matrix of differences on which we would compute the row-wise...
    # L2-norm. this method might be less efficient but was what initially explained so I instead calculated the norm in the loop as well

"""Nowe we have our C_gap and Xnorm arrays ready to use"""
norm_diag = np.diag(difference_norm_array)
cost = 1/(N*(N-1)) * np.ones((1, N**2)) @ ((1 - C_Gap_Array @ norm_diag).T + np.linalg.inv(difference_norm_array) @ C_Gap_Array.T)








