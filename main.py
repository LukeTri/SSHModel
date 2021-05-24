import math
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def Hamiltonian(v, w, n):
    x = []
    y = []
    y2 = []
    dx = []
    dy = []
    for i in range(0, n+1):
        k=2 * math.pi * i / n - math.pi
        x.append(k)
        z = math.sqrt(v*v + w*w + 2*v*w*math.cos(k))
        y.append(z)
        y2.append(-z)
        dx.append(v + w * math.cos(k))
        dy.append(w*math.sin(k))

    Hper = np.zeros((2*n,2*n))
    for i in range(0, n):
        Hper[2*i][2*i+1] = v
        Hper[2*i+1][2*i] = v
    for i in range(0, n-1):
        Hper[2*i+1][2*i+2] = w
        Hper[2*i+2][2*i+1] = w
    Haper = Hper.copy()
    Hper[0][2*n-1] = w
    Hper[2*n-1][0] = w

    Eigaper = np.linalg.eig(Haper)[0]
    Eigper = np.linalg.eig(Hper)[0]

    print(Eigaper)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.plot(x, y2, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)
    x = np.linspace(-math.pi, math.pi)
    plt.plot(x,0*x + Eigaper.max(), color = 'orange')
    plt.plot(x, 0*x + np.array([j for j in Eigaper if j > 0]).min(),color = 'red')
    plt.plot(x, 0 * x + np.array([j for j in Eigaper if j < 0]).max(), color='pink')
    plt.plot(x,0*x + Eigaper.min(), color = 'purple')

    plt.ylim(-2, 2)
    plt.xlim(-math.pi, math.pi)
    plt.xticks([-math.pi,0,math.pi])
    plt.axhline(0)
    plt.axvline(0)

    plt.xlabel('k')
    plt.ylabel('Energy')
    plt.title('Dispersion Relations')

    plt.subplot(122)

    plt.plot(dx, dy, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.xlabel('d_x')
    plt.ylabel('d_y')
    plt.axhline(0)
    plt.axvline(0)

    plt.show()


if __name__ == '__main__':
    Hamiltonian(0.5, .25,10)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
