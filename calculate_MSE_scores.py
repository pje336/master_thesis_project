import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
filepath = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/"
filename = "mse_ncc.txt"
with open(filepath + filename) as f:
    lines = f.readlines()

# mse = np.zeros((int(len(lines)/2),2))
#
# for i, line in enumerate(lines):
#     if i % 2 != 0:
#         # print(int(i/2))
#         line = line.replace("\n","").replace("[","").replace("]","").split(" ")
#         mse[int(i / 2)][0] = line[0]
#
#
#         for value in line[1:]:
#             if len(value) > 3:
#                 mse[int(i/2)][1] = value
#
#
#
# print(np.sort(mse[:,1]))



df = pd.read_csv(filepath + filename, sep = ";")
serie = df["Scan_key"].str.split(pat = "'", expand=True)
df[["MSE"]] = df[["MSE"]].astype(float)
df[["NCC"]] = df[["NCC"]].astype(float)

serie[5] = serie[5].astype(float)
serie[7] = serie[7].astype(float)
df["diff"] = abs(serie[5]-serie[7])

means = df[["MSE","NCC","diff"]].groupby("diff").mean()
std = df[["MSE","NCC","diff"]].groupby("diff").std()
min = df[["MSE","NCC","diff"]].groupby("diff").min()
max = df[["MSE","NCC","diff"]].groupby("diff").max()


print(means.index)

plt.errorbar(means.index, means["MSE"] ,yerr = std["MSE"], linestyle='', marker = "o")
plt.plot(means.index,min["MSE"], "x")
plt.plot(means.index,max["MSE"], "x")

plt.xlabel("Absolute phase difference")
plt.ylabel("mean MSE with std")

plt.show()
plt.errorbar(means.index, means["NCC"] ,yerr = std["NCC"], linestyle='', marker = "o")
plt.plot(means.index,min["NCC"], "x")
plt.plot(means.index,max["NCC"], "x")
plt.xlabel("Absolute phase difference")
plt.ylabel("mean NCC with std")

plt.show()
