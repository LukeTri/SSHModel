# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
import matplotlib.pyplot as plt

def LargeHamiltonian(N, v, w, u, t, o):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for n in range(0, N):
        for m in range(0, N):
            idxA = 2*N*n + 2*m
            idxB = 2*N*n + 2*m+1
            idxNN1 = 2 * N * ((n - 1) % N) + 2 * m + 1
            idxNN2 = 2 * N * ((n - 1) % N) + 2 * ((m + 1) % N) + 1

            idxNNNl = 2 * N * n + 2 * ((m - 1) % N)
            idxNNNr = 2 * N * n + 2 * ((m + 1) % N)
            idxNNNul = 2 * N * ((n - 1) % N) + 2 * m
            idxNNNur = 2 * N * ((n - 1) % N) + 2 * ((m + 1) % N)
            idxNNNll = 2 * N * ((n + 1) % N) + 2 * ((m - 1) % N)
            idxNNNlr = 2 * N * ((n + 1) % N) + 2 * m

            aH[idxA][idxA] = o
            aH[idxA + 1][idxA + 1] = -o

            # NN pairs
            aH[idxA][idxA + 1] = v
            aH[idxA + 1][idxA] = v

            aH[idxA][idxNN1] = w
            aH[idxNN1][idxA] = w

            aH[idxA][idxNN2] = u
            aH[idxNN2][idxA] = u

            # NNN same row
            aH[idxA][idxNNNr] = t*-1j
            aH[idxA + 1][idxNNNr + 1] = t*1j

            aH[idxA][idxNNNl] = t*1j
            aH[idxA + 1][idxNNNl + 1] = t*-1j

            # NNN above row
            aH[idxA][idxNNNul] = t*-1j
            aH[idxA + 1][idxNNNul + 1] = t*1j

            aH[idxA][idxNNNur] = t*1j
            aH[idxA + 1][idxNNNur + 1] = t*-1j

            # NNN below row
            aH[idxA][idxNNNll] = t*-1j
            aH[idxA + 1][idxNNNll+1] = t*1j

            aH[idxA][idxNNNlr] = t*1j
            aH[idxA + 1][idxNNNlr + 1] = t*-1j
    return np.linalg.eigh(aH)


def aperHamiltonian2(N, v, w, u, t, o):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for n in range(0, N):
        for m in range(0, N):
            idxA = 2*N*n + 2*m
            idxB = 2*N*n + 2*m+1
            idxNN1 = 2 * N * (n - 1) + 2 * m + 1
            idxNN2 = 2 * N * (n - 1) + 2 * (m + 1) + 1

            idxNNNl = 2 * N * n + 2 * ((m - 1) % N)
            idxNNNr = 2 * N * n + 2 * ((m + 1) % N)
            idxNNNul = 2 * N * ((n - 1) % N) + 2 * m
            idxNNNur = 2 * N * ((n - 1) % N) + 2 * ((m + 1) % N)
            idxNNNll = 2 * N * ((n + 1) % N) + 2 * ((m - 1) % N)
            idxNNNlr = 2 * N * ((n + 1) % N) + 2 * m

            aH[idxA][idxA] = o
            aH[idxA + 1][idxA + 1] = -o

            # NN pairs
            aH[idxA][idxA + 1] = v
            aH[idxA + 1][idxA] = v

            if n != 0:
                aH[idxA][idxNN1] = w
                aH[idxNN1][idxA] = w
                if m != N-1:
                    aH[idxA][idxNN2] = u
                    aH[idxNN2][idxA] = u

            # NNN same row
            if m != N-1:
                aH[idxA][idxNNNr] = t*-1j
                aH[idxA + 1][idxNNNr + 1] = t*1j

            if m != 0:
                aH[idxA][idxNNNl] = t*1j
                aH[idxA + 1][idxNNNl + 1] = t*-1j

            # NNN above row
            if n != 0:
                aH[idxA][idxNNNul] = t*-1j
                aH[idxA + 1][idxNNNul + 1] = t*1j

                if m != N-1:
                    aH[idxA][idxNNNur] = t*1j
                    aH[idxA + 1][idxNNNur + 1] = t*-1j

            # NNN below row
            if n != N-1:
                if m != 0:
                    aH[idxA][idxNNNll] = t*-1j
                    aH[idxA + 1][idxNNNll+1] = t*1j

                aH[idxA][idxNNNlr] = t*1j
                aH[idxA + 1][idxNNNlr + 1] = t*-1j
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

    plt.plot(val[N * N:], 'o')
    # plt.show()
    components = list(range(0, N))
    # breaks eigenvectors into groups
    groups = {}
    c = 0
    prindex = 0
    for i in range(0, len(val)):
        if c != len(components) - 1:
            if (val[i] - components[c]) ** 2 > (val[i] - components[c + 1]) ** 2:
                groups[c] = vec[:, prindex:i]
                prindex = i
                c = c + 1
    groups[c] = vec[:, prindex:]

    # finds P_j * Y * P_j
    for i in range(0, len(components)):
        pj = np.zeros((2 * N * N, 2 * N * N))
        arr = groups.get(i)
        for j in range(0, arr.shape[1]):
            pj = pj + np.outer(arr[:, j], arr[:, j])
        mfinal = np.matmul(np.matmul(pj, Y), pj)
        eigs = np.linalg.eigh(mfinal)[1][:, -1]

        # plots eigenvectors in 2-d graph
        x = np.linspace(0, N - 1, N)
        y = np.linspace(0, N - 1, N)
        Xx, Yy = np.meshgrid(x, y)
        Z = np.zeros((N, N))
        for n in range(0, N):
            for m in range(0, N):
                Z[n][m] = np.linalg.norm(eigs[int(2 * n * N + 2 * m)]) + np.linalg.norm(eigs[int(2 * N * n + 2 * m + 1)])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(Xx, Yy, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('surface')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(60, 35)
        plt.show()


def ChernMarker(N, L, H):

    X = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    Y = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for i in range(0, N):
        for j in range(0, N):
            X[2 * N * i + 2 * j][2 * N * i + 2 * j] = j - N//2
            Y[2 * N * i + 2 * j][2 * N * i + 2 * j] = i - N//2

            X[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = j - N//2
            Y[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = i - N//2
    # for k in range(0,2*N*N):
        # print(X[k])

    XL = np.zeros((2 * N * N, 2 * N * N), dtype='complex')

    for i in range(0, N):
        for j in range(0, N):
            if N // 2 - L < i < N // 2 + L and N // 2 - L < j < N // 2 + L:
                # print(i,j)
                XL[2 * N * i + 2 * j][2 * N * i + 2 * j] = 1
                XL[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = 1
    # for k in range(0,2*N*N):
        # print(XL[k])



    P = H @ H.conj().T
    matr = 1j * XL @ ((P @ X @ P) @ (P @ Y @ P) - (P @ Y @ P) @ (P @ X @ P)) @ XL

    return np.trace(matr) * math.pi / L / L / 2


if __name__ == '__main__':
    N = 22
    # print(temp)
    for j in range(23, 24):

        for M in range(0,10):
            if j % 2 == 1:
                print(j,M)
                d, f = aperHamiltonian2(j, 1, 1, 1, 1, M)
                f = f[:, d < 0]
                print("Aperiodic energy gap")
                print(min(np.absolute(d)))
                E, temp = LargeHamiltonian(j, 1, 1, 1, 1, M)
                temp = temp[:, E < 0]
                # plt.plot(E,'o')
                # plt.show()
                for i in range(j//2,j//2+1):
                    print("Chern Marker")
                    print(ChernMarker(j, i, temp))
    print("Bulk-boundary correspondence!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
