import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem, RBFSampler
from matplotlib import pyplot
import seaborn as sns; sns.set()

import math
import random
import numpy as np

def vector_pinv(vector, tol = -1):
    if tol == -1:
        tol = np.finfo(float).eps*np.size(vector)*np.max(vector)
    return np.array([1/x if x > tol else x for x in np.nditer(vector)])

def dot_kernel(x1,x2):
    return np.dot(x1, x2.T)

def rbf_kernel(x1,x2,gamma = 0.05):
    diff = x1-x2
    return math.exp(-gamma*dot_kernel(diff,diff))

def svd(x,**kw):
    return np.linalg.svd(x,**kw)

def construct_kernel_matrix(X1,X2,kernel=dot_kernel):
    return np.matrix(np.array([[kernel(xi,xj) for xj in X2.T] for xi in X1.T]))

def one_shot_nystrom(X,m, k = -1, kernel_func = rbf_kernel):
    "Returns the eigenvectors and eigenvalues of the approximated "
    "kernel matrix by sampling M datapoints from X and applying the one-shot Nystrom method"
    D,N = X.shape #dataset X is of size N and D dimensions
    sample = random.sample(range(N),m)  # choose m out of N for sample
    W = X[:,sample] #extract sample matrix W
    #sample by sample kernel matrix (MxM kernel matrix)
    Kw = construct_kernel_matrix(W,W, kernel = kernel_func) #W.T*W for dot product

    #whole  by sample kernel matrix (NxM truncated kernel matrix)
    C  = construct_kernel_matrix(X,W, kernel = kernel_func) #X.T*W


    U_kw, S_kw, Vt_kw = svd(Kw,full_matrices = False)  #decompose Kw
    S_kw_pinv = np.diag(vector_pinv(S_kw)) #compute the moore-penrose pseudoinverse of S_kw

    #compute the building block of Knys, the G matrix = C * V_w * S_w^-1
    #where Knys = G*G.T = C*pinv(Kw)*C.T
    #note that V_w = V_kw = Vt_kw.T
    #and S_w = sqrt(S_kw)
    #so S_w^-1 = sqrt(pinv(S_kw)) = sqrt(S_kw_pinv) computed by the previous code line
    G = C*Vt_kw.T*np.sqrt(S_kw_pinv)

    U_g, S_g, Vt_g = svd(G, full_matrices=False)
    Vnys = G * Vt_g.T * np.diag(vector_pinv(S_g))   #Vnys = U_g = G*V_g*S^-1
    #four ways to construct the kernel matrix now
    #Knys1 = G*G.T
    #Knys2 = C*Vt_kw.T*S_kw_pinv*U_kw.T*C.T     # Knys = C*pinv(Kw)*C.T ,,, pinv(Kw) = V_kw* S_kw^-1 * U_kw.T
    #Knys3 = Vnys*np.diag(S_g*S_g)*Vnys.T       # Knys = Vnys * Sg^2 * Vnys.T
    #Knys4 = U_g*np.diag(S_g*S_g)*U_g.T
    if k > 0 :
        S_nys_k = S_g[:k]
        return Vnys[:,:k],S_nys_k*S_nys_k
    return Vnys, S_g*S_g #,Knys1,Knys2,Knys3,Knys4


def circle(n):
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.normal(10, 1.0, n)

    return np.array(
        (
            radii * np.cos(angles),
            radii * np.sin(angles),
        ),
    ).T


def main():
    data = pd.DataFrame(circle(3000), columns=['x', 'y'])
    ns = Nystroem(n_components=3000, gamma=0.15)
    ns.fit(data)

    fullsample = pd.DataFrame([
            (x, y)
            for x in np.around(
                np.linspace(-20.0, 20.0, 128),
                decimals=2,
            )
            for y in np.around(
                np.linspace(-20.0, 20.0, 128),
                decimals=2,
            )
        ],
        columns=['x', 'y'],
    )

    transformed = ns.transform(fullsample)

    fullsample['c'] = pd.Series(transformed[:, 0])
    sns.heatmap(fullsample.pivot('x', 'y', 'c'), xticklabels=32, yticklabels=32)
    pyplot.show()

if __name__ == '__main__':
    main()


