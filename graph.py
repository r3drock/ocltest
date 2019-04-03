#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/home/r3drock/g/ocltest/b.csv')
X = df[['Time']]
X = np.array(X)

plt.plot(X)
plt.ylabel('Time')

plt.savefig('c.png', dpi=1000)
