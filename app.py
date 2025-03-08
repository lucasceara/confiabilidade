import numpy as np
import itertools
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# 🚀 **1. Carregar os Pesos Treinados e o MinMaxScaler**
@st.cache_data
def carregar_modelo():
    with open("pesos_treinados.pkl", "rb") as f:
        pesos_treinados = pickle.load(f)

    with open("scaler_treinado.pkl", "rb") as f:
        scaler = pickle.load(f)

    return pesos_treinados, scaler

pesos_treinados, scaler = carregar_modelo()

# 🚀 **2. Função da Rede Neural**
def neural_network(X, weights):
    W1 = weights[:X.shape[1] * 8].reshape((X.shape[1], 8))
    b1 = weights[X.shape[1] * 8:X.shape[1] * 8 + 8]

    W2_start = X.shape[1] * 8 + 8
    W2_end = W2_start + 8 * 6
    W2 = weights[W2_start:W2_end].reshape((8, 6))
    b2 = weights[W2_end:W2_end + 6]

    W3_start = W2_end + 6
    W3_end = W3_start + 6 * 1
    W3 = weights[W3_start:W3_end].reshape((6, 1))
    b3 = weights[W3_end]

    hidden1 = np.tanh(np.dot(X, W1) + b1)
    hidden2 = np.tanh(np.dot(hidden1, W2) + b2)
    output = np.dot(hidden2, W3) + b3
    return output.flatten()

# 🚀 **3. Interface no Streamlit**
st.title("Análise de Deformação - Método de Rosenblueth")

st.subheader("Insira os valores médios das variáveis do pavimento")

# Inputs dos valores médios
E_r = st.number_input("Módulo de resiliência do revestimento (MPa)", min_value=0.0, value=300.0)
h_r = st.number_input("Espessura do revestimento (m)", min_value=0.0, value=0.2)
E_b = st.number_input("Módulo de resiliência da base (MPa)", min_value=0.0, value=200.0)
h_b = st.number_input("Espessura da base (m)", min_value=0.0, value=0.3)
E_sb = st.number_input("Módulo de resiliência da sub-base (MPa)", min_value=0.0, value=150.0)
h_sb = st.number_input("Espessura da sub-base (m)", min_value=0.0, value=0.3)
E_s = st.number_input("Módulo de resiliência do subleito (MPa)", min_value=0.0, value=100.0)

# Inputs dos coeficientes de variação
st.subheader("Coeficientes de Variação (em decimal, ex: 0.10 para 10%)")

cv_E_r = st.number_input("Coef. de variação do Módulo de Resiliência do Revestimento", min_value=0.0, max_value=1.0, value=0.20)
cv_h_r = st.number_input("Coef. de variação da Espessura do Revestimento", min_value=0.0, max_value=1.0, value=0.10)
cv_E_b = st.number_input("Coef. de variação do Módulo de Resiliência da Base", min_value=0.0, max_value=1.0, value=0.20)
cv_h_b = st.number_input("Coef. de variação da Espessura da Base", min_value=0.0, max_value=1.0, value=0.10)
cv_E_sb = st.number_input("Coef. de variação do Módulo de Resiliência da Sub-base", min_value=0.0, max_value=1.0, value=0.20)
cv_h_sb = st.number_input("Coef. de variação da Espessura da Sub-base", min_value=0.0, max_value=1.0, value=0.10)
cv_E_s = st.number_input("Coef. de variação do Módulo de Resiliência do Subleito", min_value=0.0, max_value=1.0, value=0.20)

# Botão para executar a análise
if st.button("Calcular Deformação"):
    valores_medios = [E_r, h_r, E_b, h_b, E_sb, h_sb, E_s]
    cv_valores = [cv_E_r, cv_h_r, cv_E_b, cv_h_b, cv_E_sb, cv_h_sb, cv_E_s]

    # 🚀 **4. Gerar os 128 Cenários (+CV e -CV)**
    cenarios = []
    for comb in itertools.product([-1, 1], repeat=7):
        novo_cenario = [valores_medios[i] * (1 + comb[i] * cv_valores[i]) for i in range(7)]
        cenarios.append(novo_cenario)

    # 🚀 **5. Normalizar os Cenários**
    cenarios_scaled = scaler.transform(cenarios)

    # 🚀 **6. Rodar a Rede Neural para os 128 Cenários**
    deformacoes = neural_network(cenarios_scaled, pesos_treinados)

    # 🚀 **7. Converter as Deformações para Valores Positivos**
    deformacoes_abs = np.abs(deformacoes)

    # 🚀 **8. Calcular Média e Desvio Padrão**
    media_deformacao = np.mean(deformacoes_abs)
    desvio_padrao_deformacao = np.std(deformacoes_abs)

    # 🚀 **9. Calcular Deformação Probabilística (95%)**
    deformacao_95 = media_deformacao + 1.645 * desvio_padrao_deformacao

    # 🚀 **10. Garantir que a Deformação Probabilística seja Maior**
    deformacao_95_final = max(media_deformacao, deformacao_95)

    # 🚀 **11. Exibir Resultados**
    st.subheader("Resultados da Análise")
    st.write(f"**Deformação Determinística:** {media_deformacao:.10f} m/m")
    st.write(f"**Desvio Padrão da Deformação:** {desvio_padrao_deformacao:.10f} m/m")
    st.write(f"**Deformação Probabilística (95% de confiança):** {deformacao_95_final:.10f} m/m")
