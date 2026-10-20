import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.subheader("Ceci est une application qui permet de faire des choses merveilleuses.")

st.latex("P_n(x)=\sum_{k=0}^n\dfrac{f^{(k)}(0)}{k!}x^k")
