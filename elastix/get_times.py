
import numpy as np
from decimal import Decimal
times = []
with open("C:/Users/pje33/Downloads/time.txt", "r") as f:
    text = f.readlines()[4::3]
    for t in text:
        times.append(float(t.split(":")[-1][:-2]))

print("Model",'%.3E' % Decimal(np.mean(times)),"$\pm$",'%.3E' % Decimal(np.std(times)))