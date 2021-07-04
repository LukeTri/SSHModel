# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
from scipy.optimize import curve_fit
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
    return aH


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
    return aH

def vecPlot(vec,N):
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)
    Xx, Yy = np.meshgrid(x, y)
    Z = np.zeros((N, N))
    for n in range(0, N):
        for m in range(0, N):
            Z[n][m] = np.linalg.norm(vec[int(2 * n * N + 2 * m)])
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
    m1 = P @ X @ P
    linalg = np.linalg.eigh(m1)
    val = linalg[0]
    vec = linalg[1]
    vec = vec[:, N*N + N:]
    val = val[N*N + N:]

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
        if i == 3:
            arr = groups.get(i)
            pj = arr @ arr.conj().T
            mfinal = pj @ Y @ pj
            v,eigs = np.linalg.eigh(mfinal)
            eigs = eigs[:,v>0.1]
            eigs = eigs[:,2]
            # plots eigenvectors in 2-d graph
            vecPlot(eigs,N)


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

def rMatrix(s,N):
    rng = np.random.default_rng(seed=s)
    arr1 = rng.random(2*N**2)
    ret = np.zeros((2*N**2,2*N**2))
    for n in range(0,2*N**2):
        ret[n][n] = (arr1[n] - .5)*2
    return ret

def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c

if __name__ == '__main__':
    EXPECTED = 0
    # print(temp)
    Y = np.zeros(3)
    X = np.zeros(3)
    for j in range(9, 15):

        for M in range(0,1):
            if j % 2 == 1:

                d, f = np.linalg.eigh(aperHamiltonian2(j, 1, 1, 1, 1, 1))
                f = f[:, d < 0]
                IPP(j,f)
                print("Aperiodic energy gap")
                print(min(np.absolute(d)))
                off = rMatrix(50,j)*.5
                M = LargeHamiltonian(j, 1, 1, 1, 1, 1)
                E, temp = np.linalg.eigh(M)
                temp = temp[:, E < 0]

                for i in range(j//3,j//3+1):
                    print(j, i)
                    print("Chern Marker")
                    res = ChernMarker(j, i, temp)
                    print(res)
                    Y[(j-9)//2] = EXPECTED-res
                    X[(j - 9) // 2] = j
    print("Bulk-boundary correspondence!")

    # logy = np.log(Y)
    # print(logy)
    # coeffs = np.polyfit(X, logy, deg=1)
    # poly = np.poly1d(coeffs)
    # yfit = lambda x: np.exp(poly(x))

    # Xnew = X
    # for w in range(33,61):
        # if w % 2 == 1:
            # Xnew = np.append(Xnew, w)
    # plt.plot(Xnew, yfit(Xnew))


    plt.plot(X,Y)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
