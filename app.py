import streamlit as st

st.title("Análise de Deformação - Método de Rosenblueth")

st.subheader("Insira os valores médios das variáveis do pavimento")

E_r = st.number_input("Módulo de resiliência do revestimento (MPa)", min_value=0.0, value=300.0)
h_r = st.number_input("Espessura do revestimento (m)", min_value=0.0, value=0.2)
E_b = st.number_input("Módulo de resiliência da base (MPa)", min_value=0.0, value=200.0)
h_b = st.number_input("Espessura da base (m)", min_value=0.0, value=0.3)
E_sb = st.number_input("Módulo de resiliência da sub-base (MPa)", min_value=0.0, value=150.0)
h_sb = st.number_input("Espessura da sub-base (m)", min_value=0.0, value=0.3)
E_s = st.number_input("Módulo de resiliência do subleito (MPa)", min_value=0.0, value=100.0)

if st.button("Calcular Deformação"):
    st.write("Aqui entra a lógica de cálculo usando a rede neural.")