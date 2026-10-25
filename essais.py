import autograd.numpy as np
from autograd import grad

# 2. Découverte d'autograd

#X=np.linspace(0,2*np.pi,1000)

sin_der=grad(np.sin)

#Y=np.vectorize(sin_der)(X)

#print(Y)  #-> bien un cosinus sur [0,2pi]

# 3. Formule de Taylor

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
    Y=np.zeros(X.shape[0])
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
    derivatives=[g(0.)]
    for i in range(1,n+1):
        g=grad(g)
        derivatives.append(g(0.))

    return taylor1(X,derivatives)

Y2=taylor2(X, np.exp, 2)

#plt.plot(X,Y2)
#plt.plot(X,1 + X + ((X**2)/2))
#plt.show()


#GROS PROBLEME ICI: le résultat du plot dépend du nombre de fois qu'on a lancé le programme ... (???)
# /\
# || Je laisse ce commentaire parce qu'il m'a pris 1h30 de débuggage avant de succomber à l'IA:
# J'utilisais np.empty pour initialiser Y, au lieu de np.zeros. Donc j'avais des résultats qui dépendaient
# de ce qu'il y a quelque part en mémoire au moment du lancement, parfois des zéros, parfois des valeurs non-nulles.
# Je pensais que np.empty allouait un espace à un tableau de valeurs vides/Nan, en fait non.


# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

#-> Pas d'erreur renvoyée, les deux courbes sont proches à e-5 ou e-8 près


# sinus au degré 10, à comparer avec np.sin(X)
Y3 = np.sin(X)
Y4 = taylor2(X, np.sin, 10)

#plt.plot(X,Y3)
#plt.plot(X,Y4)
#plt.show()

DIFF=np.absolute(Y4-Y3)
Iproche= [i for i in range(len(X)) if DIFF[i]<1e-3 ]

#print(f"La proportion du domaine où les courbes sont égales à 10^-3 près est: {len(Iproche)*100/len(X)}%.")
#-> 42%

Y5 = np.cos(X)
Y6 = taylor2(X,np.cos,0)

plt.plot(X,Y5)
plt.plot(X,Y6)
plt.show()
# -> Le degré 0 approxime la fonction par une valeur constante, ici égale à np.cos(0.)
