# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time


def LargeHamiltonian(N, v, w, u, t, o):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for n in range(0, N):
        for m in range(0, N):
            idxA = 2 * N * n + 2 * m
            idxB = 2 * N * n + 2 * m + 1
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
            aH[idxA][idxNNNr] = t * -1j
            aH[idxA + 1][idxNNNr + 1] = t * 1j

            aH[idxA][idxNNNl] = t * 1j
            aH[idxA + 1][idxNNNl + 1] = t * -1j

            # NNN above row
            aH[idxA][idxNNNul] = t * -1j
            aH[idxA + 1][idxNNNul + 1] = t * 1j

            aH[idxA][idxNNNur] = t * 1j
            aH[idxA + 1][idxNNNur + 1] = t * -1j

            # NNN below row
            aH[idxA][idxNNNll] = t * -1j
            aH[idxA + 1][idxNNNll + 1] = t * 1j

            aH[idxA][idxNNNlr] = t * 1j
            aH[idxA + 1][idxNNNlr + 1] = t * -1j
    return aH


def aperHamiltonian2(N, v, w, u, t, o):
    aH = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for n in range(0, N):
        for m in range(0, N):
            idxA = 2 * N * n + 2 * m
            idxB = 2 * N * n + 2 * m + 1
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
                if m != N - 1:
                    aH[idxA][idxNN2] = u
                    aH[idxNN2][idxA] = u

            # NNN same row
            if m != N - 1:
                aH[idxA][idxNNNr] = t * -1j
                aH[idxA + 1][idxNNNr + 1] = t * 1j

            if m != 0:
                aH[idxA][idxNNNl] = t * 1j
                aH[idxA + 1][idxNNNl + 1] = t * -1j

            # NNN above row
            if n != 0:
                aH[idxA][idxNNNul] = t * -1j
                aH[idxA + 1][idxNNNul + 1] = t * 1j

                if m != N - 1:
                    aH[idxA][idxNNNur] = t * 1j
                    aH[idxA + 1][idxNNNur + 1] = t * -1j

            # NNN below row
            if n != N - 1:
                if m != 0:
                    aH[idxA][idxNNNll] = t * -1j
                    aH[idxA + 1][idxNNNll + 1] = t * 1j

                aH[idxA][idxNNNlr] = t * 1j
                aH[idxA + 1][idxNNNlr + 1] = t * -1j
    return aH


def vecPlot(vec, N):
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)
    Xx, Yy = np.meshgrid(x, y)
    Z = np.zeros((N, N))
    for n in range(0, N):
        for m in range(0, N):
            Z[n][m] = np.linalg.norm(vec[int(2 * n * N + 2 * m)]) + np.linalg.norm(vec[int(2 * n * N + 2 * m + 1)])
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


def IPP(N, H, p, i_index,j_index):
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
    m1 = P @ X @ P
    linalg = np.linalg.eigh(m1)
    val = linalg[0]
    vec = linalg[1]
    vec = vec[:, N * N + N:]
    val = val[N * N + N:]

    # breaks eigenvectors into groups
    components = list(range(1, N))
    groups = {}
    c = 0
    prindex = 0
    for i in range(1, len(val)):
        if c != len(components) - 1:

            if (val[i] - components[c]) ** 2 > (val[i] - components[c + 1]) ** 2:
                groups[c] = vec[:, prindex:i]
                prindex = i
                c = c + 1
    groups[c] = vec[:, prindex:]


    # finds P_j * Y * P_j
    for i in range(0, len(components)):
        if i == i_index:
            arr = groups.get(i)
            pj = arr @ arr.conj().T
            mfinal = pj @ Y @ pj
            v, eigs = np.linalg.eigh(mfinal)
            eigs = eigs[:, v > 0.1]
            eigs = eigs[:, j_index]
            # plots eigenvector in 2-d graph
            if p:
                vecPlot(eigs, N)
            # Calculates localization of eigenfunction
            newArr = np.zeros((N, N))
            for n in range(0, N):
                for m in range(0, N):
                    newArr[n][m] = np.linalg.norm(eigs[2 * N * n + 2 * m]) + np.linalg.norm(eigs[2 * N * n + 2 * m + 1])
            flatX = np.sum(newArr, axis=0)

            # x-direction spread
            WeightedX = 0
            for l in range(0, N):
                WeightedX = WeightedX + l * flatX[l]
            WeightedX = WeightedX / np.sum(flatX)
            spreadX = 0
            for k in range(0, N):
                spreadX = spreadX + ((k - WeightedX) ** 2) * flatX[k]

            # y-direction spread
            flatY = np.sum(newArr, axis=1)
            WeightedY = 0
            for l in range(0, N):
                WeightedY = WeightedY + l * flatY[l]
            WeightedY = WeightedY / np.sum(flatY)
            spreadY = 0
            for k in range(0, N):
                spreadY = spreadY + ((k - WeightedY) ** 2) * flatY[k]
            return math.sqrt(spreadX ** 2 + spreadY ** 2) / (N * N)


def ChernMarker(N, L, H):
    X = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    Y = np.zeros((2 * N * N, 2 * N * N), dtype='complex')
    for i in range(0, N):
        for j in range(0, N):
            X[2 * N * i + 2 * j][2 * N * i + 2 * j] = j - N // 2
            Y[2 * N * i + 2 * j][2 * N * i + 2 * j] = i - N // 2

            X[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = j - N // 2
            Y[2 * N * i + 2 * j + 1][2 * N * i + 2 * j + 1] = i - N // 2
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


def rMatrix(s, N):
    rng = np.random.default_rng(seed=s)
    arr1 = rng.random(2 * N ** 2)
    ret = np.zeros((2 * N ** 2, 2 * N ** 2))
    for n in range(0, 2 * N ** 2):
        ret[n][n] = (arr1[n] - .5) * 2
    return ret


def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c


def get_data(EXPECTED,EXTRAPOLATE,IPPplot,v,t,o,N,seed,i_index,j_index):

    # print(temp)
    Y = np.zeros(1)
    X = np.zeros(1)
    for j in range(N, N+1):
        for M in range(0, 1):
            off = rMatrix(seed, j) * .5
            d, f = np.linalg.eigh(aperHamiltonian2(j, v, v, v, t, o) - off)
            f = f[:, d < 0]
            print("IPP eigenfunction spread")
            info_spread = IPP(j, f, IPPplot,i_index,j_index)
            print(info_spread)
            print("Aperiodic energy gap")
            info_gap = min(np.absolute(d))
            print(info_gap)

            M = LargeHamiltonian(j, v, v, v, t, o) - off
            E, temp = np.linalg.eigh(M)
            temp = temp[:, E < 0]

            for i in range(j // 3, j // 3 + 1):
                print("Chern Marker")
                info_marker = np.real(ChernMarker(j, i, temp))
                print(info_marker)
                if EXTRAPOLATE:
                    Y[(j - 9) // 2] = EXPECTED - info_marker
                    X[(j - 9) // 2] = j
    print("Bulk-boundary correspondence!")
    if EXTRAPOLATE:
        logx = np.log(X)
        logy = np.log(Y)
        coeffs = np.polyfit(logx, logy, deg=1)
        poly = np.poly1d(coeffs)
        yfit = lambda x: np.exp(poly(np.log(x)))

        Xnew = X
        for w in range(25, 35):
            if w % 2 == 1:
                Xnew = np.append(Xnew, w)
        plt.plot(Xnew, yfit(Xnew))

        plt.loglog(X, Y)
        plt.show()
    return info_spread, info_gap, info_marker


def test_data():
    file1 = open("data.txt", "a")
    for s in range(11,20):
        n = 10
        i = n//2
        j = n//2 + 1

        out1, out2, out3 = get_data(EXPECTED=1, EXTRAPOLATE=False, IPPplot=False,
                                    v=1, t=1, o=1, N=n, seed=s, i_index=i, j_index=j)
        file1.write(str(n) + ", " + str(s) + ", " + str(i) + ", " + str(j) + ", " + str(out1) + ", " +
                    str(out2) + ", " + str(out3) + "\n")
    file1.close()


test_data()
