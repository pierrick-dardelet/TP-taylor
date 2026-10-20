import autograd.numpy as np
from autograd import grad

# 2. Découverte d'autograd

X=np.linspace(0,2*np.pi,1000)

sin_der=grad(np.sin)

Y=np.vectorize(sin_der)(X)

#print(Y)  #-> bien un cosinus sur [0,2pi]

# 3.

from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
    ordre=len(derivatives)
    Y=np.empty(X.shape[0])
    for n in range(ordre):
        TermeN=(derivatives[n]/factorial(n))*(X**(n))
        Y=Y+TermeN
    
    return Y


sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

#plt.plot(X,taylor1(X,sinus10))
#plt.plot(X,np.vectorize(np.sin)(X))
#plt.show()



def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
    g=f
    derivatives=[g(0.0)]
    for i in range(1,n+1):
        g=grad(g)
        derivatives.append(g(0.0))

    return taylor1(X,derivatives)


plt.plot(X,taylor2(X, np.exp, 2))
print(taylor2(X, np.exp, 2))
plt.show()

plt.plot(X,1 + X + ((X**2)/2))
print(1 + X + ((X**2)/2))
plt.show()


#GROS PROBLEME ICI: le résultat du plot dépend du nombre de fois qu'on a lancé le programme ... (???)

# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + ((X**2)/2))



# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)
