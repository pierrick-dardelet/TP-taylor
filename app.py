import streamlit as st
import autograd.numpy as np
from autograd import grad
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

def custom(x):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2*np.sin(x)+np.cos(x/4)


FONCTIONS={"sinus":np.sin,"cosinus":np.cos,"exponentielle":np.exp,"logarithme nepérien translaté":lambda x:np.log(1+x),"super fonction":custom}



st.title("Approximation de Taylor")
with st.sidebar:
    fname=st.selectbox("Quelle fonction approximer?",FONCTIONS)

st.subheader("APPROXIMATION DE FONCTIONS")
st.write("Cet outil utilise l'approximation de Taylor, qui s'écrit à l'ordre n:")
st.latex("P_n(x)=\sum_{k=0}^n\dfrac{f^{(k)}(0)}{k!}x^k")
n=st.slider("A quel ordre?",0,10,3)
l=st.slider("Sur quelle largeur d'intervalle centré en 0?",2.,20.,12.6,0.01)

f=FONCTIONS[fname]
X=np.linspace(-l/2,l/2,400)
Y=f(X)
Y_taylor=taylor2(X,f,n)

fig,ax=plt.subplots()
ax.plot(X,Y,label='Fonction f')
ax.plot(X,Y_taylor,label='Approximation de f par Taylor')
ax.legend()
st.pyplot(fig)

# La qualité d'approximation empire à mesure qu'on s'éloigne de 0.
# Au-delà de abs(x)=1, l'approximation s'écarte de la fonction à mesure que n augmente.
# Cohérent avec le rayon de la série entière qui vaut 1.