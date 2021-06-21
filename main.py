# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math

import matplotlib.pyplot as plt


def aperHamiltonian(N, v, w,u,t,o):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for i in range(0, 2 * N * N):
        if i % 2 == 0:
            aH[i][i] = o
            aH[i + 1][i + 1] = -o
            aH[i][i + 1] = v
            aH[i + 1][i] = v
            if i / (2 * N) != 0:
                aH[i][i - 2 * N + 1] = w
                aH[i - 2 * N + 1][i] = w

                if (i - 2 * N + 3) % (2 * N) != 1:
                    aH[i][i - 2 * N + 3] = u
                    aH[i - 2 * N + 3][i] = u

            if (i + 2) % (2 * N) != 0:
                aH[i][i + 2] = t
                aH[i + 2][i] = t

                aH[i + 1][i + 3] = t
                aH[i + 3][i + 1] = t
            if i % (2 * N) != 0:
                aH[i][i - 2] = t
                aH[i - 2][i] = t

                aH[i + 1][i - 1] = t
                aH[i - 1][i + 1] = t
            if i / (2 * N) != 0:
                aH[i][i - 2 * N] = t
                aH[i - 2 * N][i] = t

                aH[i + 1][i - 2 * N + 1] = t
                aH[i - 2 * N + 1][i + 1] = t
                if (i - 2 * N + 2) % (2 * N) != 1:
                    aH[i][i - 2 * N + 2] = t
                    aH[i - 2 * N + 2][i] = t

                    aH[i + 1][i - 2 * N + 3] = t
                    aH[i - 2 * N + 3][i + 1] = t
            if math.floor(i / (2 * N)) != N - 1:
                aH[i][i + 2 * N] = t
                aH[i + 2 * N][i] = t

                aH[i + 1][i + 2 * N + 1] = t
                aH[i + 2 * N + 1][i + 1] = t
                if i % (2 * N) != 0:
                    aH[i][i + 2 * N - 2] = t
                    aH[i + 2 * N - 2][i] = t

                    aH[i + 1][i + 2 * N - 1] = t
                    aH[i + 2 * N - 1][i + 1] = t
    return np.linalg.eigh(aH)

def IPP(N, H):
    # creates position operators X,Y
    X = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    Y = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for i in range(0, N):
        for j in range(0, N):
            X[2 * N * i + 2 * j][2 * N * i + 2 * j] = j
            Y[2 * N * i + 2 * j][2 * N * i + 2 * j] = i

            X[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = j
            Y[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = i

    # finds eigenvectors/values of P * X * P
    P = H @ H.conj().T
    m1 = np.matmul(np.matmul(P, X), P)
    linalg = np.linalg.eigh(m1)
    val = linalg[0]
    vec = linalg[1]

    plt.plot(val[N*N:], 'o')
    plt.show()
    components = list(range(0, N))
    # breaks eigenvectors into groups
    groups = {}
    c = 0
    prindex = 0
    for i in range(0, len(val)):
        if c != len(components) - 1:
            if (val[i] - components[c]) ** 2 > (val[i] - components[c + 1]) ** 2:
                groups[c] = vec[:,prindex:i]
                prindex = i
                c = c + 1
    groups[c] = vec[:,prindex:]

    # finds P_j * Y * P_j
    for i in range(0, len(components)):
        pj = np.zeros((2*N*N, 2*N*N))
        arr = groups.get(i)
        for j in range(0, arr.shape[1]):
            pj = pj + np.outer(arr[:,j], arr[:,j])
        mfinal = np.matmul(np.matmul(pj, Y), pj)
        plt.plot(np.linalg.eigh(mfinal)[1][:,-1])
        plt.show()
        assert False


if __name__ == '__main__':
    temp = aperHamiltonian(5,1,1,1,.1,1)[1]
    temp = temp[:,:temp.shape[1]//2]
    IPP(5, temp)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
