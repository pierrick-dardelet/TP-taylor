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
    log10seuil=st.slider("Quel seuil de tolérance en précision en log10?",-6.,-2.,-3.,step=0.1)

st.subheader("APPROXIMATION DE FONCTIONS")
st.write("Cet outil utilise l'approximation de Taylor, qui s'écrit à l'ordre n:")
st.latex("P_n(x)=\sum_{k=0}^n\dfrac{f^{(k)}(0)}{k!}x^k")
n=st.slider("A quel ordre?",0,10,3,key='ordre')
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



#5. Qualité d'approximation

seuil=np.exp(log10seuil*np.log(10))

def taux_coincidence(Ya,Yb,seuil):
    DIFF=np.absolute(Ya-Yb)
    I_PROCHE= [i for i in range(len(X)) if DIFF[i]<seuil ]
    return len(I_PROCHE)/len(Ya)

st.write(f"Avec le seuil choisi à {seuil:.2g}, l'approximation coïncide avec la fonction sur {round(taux_coincidence(Y,Y_taylor,seuil)*100,3)}% du domaine affiché.")

# Cf tests pour 5.1 dans essais.py: nan<1e-3 vaut False
# Donc tout le domaine affiché où la fonction n'est pas définie compte comme "ne coïncidant pas" (ce qui est raisonnable) et abaisse le taux renvoyé.

case=st.checkbox("Afficher l'erreur")

if case:
    fig2,ax2=plt.subplots()
    ax2.plot(X,np.absolute(Y_taylor-Y),label="écart |f-Pn|")
    ax2.semilogy()
    ax2.legend()
    st.pyplot(fig2)

# 6. Comparer plusieurs ordres

st.subheader('COMPARAISON DE PLUSIEURS ORDRES')
bouton = st.button("Ajouter cet ordre à la comparaison")
reset = st.button("Réinitialisation")

if "A_comparer" not in st.session_state.keys():
    st.session_state['A_comparer']=[]

if bouton and n not in st.session_state['A_comparer']:
    st.session_state['A_comparer'].append(n)

if reset:
    st.session_state['A_comparer']=[]

#print(st.session_state) #-> Evite bien les doublons et persiste bien entre deux exécutions.

fig3,ax3=plt.subplots()
ax3.plot(X,Y,label='f')
for ordre in st.session_state['A_comparer']:
    ax3.plot(X,taylor2(X,f,ordre),label=f'P_{ordre}')
ax3.legend()
st.pyplot(fig3)
