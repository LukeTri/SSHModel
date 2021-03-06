import numpy as np
import math
import cmath
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Hamiltonian(N=30, v=1.0, w=1.0, show=False, u=1.0, t=1.0):
    eig = np.zeros((N, N))
    matr = np.zeros((N, N, 2, 2), dtype='complex')
    for kx in range(1, N + 1):
        for ky in range(1, N + 1):
            H = -v * cmath.exp(-1j * ((2 * math.pi) / N) * (ky)) - w * cmath.exp(
                -1j * ((2 * math.pi) / N) * (-kx * (math.sqrt(3) / 2) - ky / 2)) - u * cmath.exp(
                -1j * ((2 * math.pi) / N) * (kx * (math.sqrt(3) / 2) - ky / 2))
            z = -2 * t * (math.sin((2 * math.pi) / N * math.sqrt(3) * kx) + math.sin(
                (2 * math.pi) / N * (-kx * (math.sqrt(3) / 2) - ky * 3 / 2)) + math.sin(
                (2 * math.pi) / N * (-kx * (math.sqrt(3) / 2) + ky * 3 / 2)))
            matr[kx - 1][ky - 1] = np.array([[z, H], [np.conj(H), -z]])
            eig[kx - 1][ky - 1] = \
                cmath.polar(np.linalg.eig(matr[kx - 1][ky - 1])[0][0] - np.linalg.eig(matr[kx - 1][ky - 1])[0][1])[0]

    if show:
        eig[0][0] = 0
        plt.imshow(eig, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return matr


out = Hamiltonian(100, 2, 5, False, 1, 10.0)


# v is amplitude between red and blacks, u is between same color
#aperHamiltonian constructs the 2*n by 2*n aperiodic hamiltonian

def aperHamiltonian(N, v, u):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for i in range(0, 2 * N * N):
        if i % 2 == 0:
            aH[i][i + 1] = v
            aH[i + 1][i] = v
            if i / (2 * N) != 0:
                aH[i][i - 2 * N + 1] = v
                aH[i - 2 * N + 1][i] = v

                if (i - 2 * N + 3) % (2 * N) != 1:
                    aH[i][i - 2 * N + 3] = v
                    aH[i - 2 * N + 3][i] = v

            if (i + 2) % (2 * N) != 0:
                aH[i][i + 2] = u
                aH[i + 2][i] = u

                aH[i + 1][i + 3] = u
                aH[i + 3][i + 1] = u
            if i % (2 * N) != 0:
                aH[i][i - 2] = u
                aH[i - 2][i] = u

                aH[i + 1][i - 1] = u
                aH[i - 1][i + 1] = u
            if i / (2 * N) != 0:
                aH[i][i - 2 * N] = u
                aH[i - 2 * N][i] = u

                aH[i + 1][i - 2 * N + 1] = u
                aH[i - 2 * N + 1][i + 1] = u
                if (i - 2 * N + 2) % (2 * N) != 1:
                    aH[i][i - 2 * N + 2] = u
                    aH[i - 2 * N + 2][i] = u

                    aH[i + 1][i - 2 * N + 3] = u
                    aH[i - 2 * N + 3][i + 1] = u
            if i / (2 * N) != N - 1:
                aH[i][i + 2 * N] = u
                aH[i + 2 * N][i] = u

                aH[i + 1][i + 2 * N + 1] = u
                aH[i + 2 * N + 1][i + 1] = u
                if i % (2 * N) != 0:
                    aH[i][i + 2 * N - 2] = u
                    aH[i + 2 * N - 2][i] = u

                    aH[i + 1][i + 2 * N - 1] = u
                    aH[i + 2 * N - 1][i + 1] = u
    return aH


def getVec(kx, ky, c, N, v):
    vec = np.zeros((2 * N * N), dtype='complex')
    vec[2 * ky * N + 2 * kx + c] = cmath.exp(1j * (ky * v)) + cmath.exp(
        1j * (kx * math.sqrt(3) / 2 + ky * -v / 2)) + cmath.exp(1j * (kx * math.sqrt(3) / 2 + ky * -v / 2))
    return vec


# aperBerry method gives NxN array of the B-M hamiltonians for the aperiodic case (same form as Hamiltonian function)

def aperBerry(N, v, w):
    matr = np.zeros((N, N, 2, 2), dtype='complex')
    aH = aperHamiltonian(N, v, w)
    for kx in range(0, N):
        for ky in range(0, N):
            temp = np.zeros((2, 2), dtype='complex')
            for A in range(0, 2):
                for B in range(0, 2):
                    temp[A][B] = np.dot(np.matmul(getVec(kx, ky, A, N, v), aH), getVec(kx, ky, B, N, v))
            matr[kx][ky] = temp
    return matr


def Berry(N, v, w, u, t):
    matr = aperBerry(N, v, w)
    pos = np.zeros((N, N, 2))

    dat = np.zeros((N, N, 2 * N * N), dtype='complex')

    for kx in range(1, N + 1):
        for ky in range(1, N + 1):

            vec = np.linalg.eig(matr[kx - 1][ky - 1])[1][:, np.argmax(np.linalg.eig(matr[kx - 1][ky - 1])[0])]
            # vec = vec/vec[0]
            # vec = np.array([1,matr[kx-1][ky-1][0][1]/cmath.polar(matr[kx-1][ky-1][0][1])[0]])

            for n in range(0, N):
                for m in range(0, N):
                    dat[kx - 1][ky - 1][(2 * n) * N + 2 * m] = cmath.exp(((2 * math.pi) / N) * (kx * n + ky * m)) * vec[
                        0]
                    dat[kx - 1][ky - 1][(2 * n) * N + 2 * m + 1] = cmath.exp(((2 * math.pi) / N) * (kx * n + ky * m)) * \
                                                                   vec[1]
            dat[kx - 1][ky - 1] = dat[kx - 1][ky - 1] / np.linalg.norm(dat[kx - 1][ky - 1])
    sumterm = 0
    for n in range(0, N):
        for m in range(0, N):
            sumterm = sumterm - cmath.polar(np.vdot(dat[n][m], dat[(n + 1) % N][m]) * np.vdot(dat[(n + 1) % N][m],
                                                                                              dat[(n + 1) % N][(
                                                                                                                       m + 1) % N]) * np.vdot(
                dat[(n + 1) % N][(m + 1) % N], dat[n][(m + 1) % N]) * np.vdot(dat[n][(m + 1) % N], dat[n][m]))[1]
    print(sumterm / (2 * math.pi))


Berry(4, 2, 7, 1, 0.0)
