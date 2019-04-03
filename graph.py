#!/usr/bin/python

#plt.plot(X)
#plt.plot(Y)
#plt.ylabel('Time')

#plt.savefig('c.png', dpi=1000)

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/home/r3drock/g/ocltest/b.csv')

W = df[['Index']]
W = np.array(W)
X = df[['Time']]
X = np.array(X)
Y = df[['TimePerElement']]
Y = np.array(Y)

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 0
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                        offset=(offset, 0))

par2.axis["right"].toggle(all=True)

host.set_xlabel("Size")
host.set_ylabel("Time")
par2.set_ylabel("Time")

p1, = host.plot(W, X, label="all")
p2, = par1.plot(W, Y, label="element")

par2.set_ylim(0, Y[0])

host.legend()

host.axis["left"].label.set_color(p1.get_color())

plt.draw()

plt.savefig("c.png", dpi = 400)
