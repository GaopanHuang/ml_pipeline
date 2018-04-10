#coding:utf-8
#plot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
#scatter: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
#bar: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar

import matplotlib.pyplot as plt
import numpy as np

#点到点的折线
x = np.arange(0,2*np.pi,0.5)
y = np.sin(x)
z = np.cos(x)
plt.plot(x,y,label='sin(x)')
plt.plot(x,z,label='cos(x)')
plt.legend()
##plt.legend(loc=(1.1, .5))
##plt.subplots_adjust(right=.75)
plt.show()

#散点图
x = np.random.rand(10)
y = np.random.rand(10)
z = np.sqrt(x**2 + y**2)

plt.subplot(121)
plt.scatter(x, y, s=80, c=z, marker=">")
plt.subplot(122)
plt.scatter(x, y, s=80, c=z, marker=(5, 1))
plt.show()
