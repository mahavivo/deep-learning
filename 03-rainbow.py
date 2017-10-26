import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
u = np.linspace(-1,1,100)
x,y = np.meshgrid(u,u)     # 网格坐标生成函数
z = np.abs((1-x**2-y**2))**0.5
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,z,rstride=4,cstride=4,cmap='rainbow')
plt.show()