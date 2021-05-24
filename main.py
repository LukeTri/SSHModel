import math

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


    plt.figure(1)
    plt.subplot(121)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.plot(x, y2, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)

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
    Hamiltonian(.5, .5,4)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
