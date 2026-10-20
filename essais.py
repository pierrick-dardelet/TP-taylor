import autograd.numpy as np
from autograd import grad

# 2. Découverte d'autograd

X=np.linspace(0,2*np.pi,1000)

sin_der=grad(np.sin)

Y=np.vectorize(sin_der)(X)

print(Y)

# 3.

from math import factorial

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
