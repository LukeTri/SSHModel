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



    plt.figure(1)
    plt.subplot(131)
    plt.plot(x, y, color='green', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.plot(x, y2, color='green', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=5)




    plt.ylim(-2, 2)
    plt.xlim(-math.pi, math.pi)
    plt.xticks([-math.pi,0,math.pi])
    plt.axhline(0)
    plt.axvline(0)

    plt.xlabel('k')
    plt.ylabel('Energy')
    plt.title('Momentum-Space Hamiltonian')

    plt.subplot(132)
    Eigper = np.sort(Eigper)
    Eigper = np.round(Eigper,2)
    Eigper = np.unique(Eigper)

    arr1 = Eigper[0:len(Eigper) / 2]
    arr2 = Eigper[len(Eigper)/2:]

    farr1 = np.flip(arr1)
    farr1 = farr1[0:len(farr1) - 1]

    farr2 = np.flip(arr2)
    farr2 = farr2[1:len(farr2)]


    arr1 = np.append(farr1, arr1)
    arr2 = np.append(arr2, farr2)

    plt.plot(x, y, color='green', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.plot(x, y2, color='green', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=5)

    plt.ylim(-2, 2)
    plt.xlim(-math.pi, math.pi)
    plt.xticks([-math.pi,0,math.pi])
    plt.axhline(0)
    plt.axvline(0)


    plt.xlabel('k')
    plt.ylabel('Energy')
    plt.title('Standard Hamiltonian')


    plt.subplot(133)

    plt.plot(dx, dy, color='blue', linewidth=1,
             marker='o', markerfacecolor='black', markersize=5)

    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.xlabel('d_x')
    plt.ylabel('d_y')
    plt.axhline(0)
    plt.axvline(0)

    plt.show()


if __name__ == '__main__':
    Hamiltonian(0.125, .25,6)
