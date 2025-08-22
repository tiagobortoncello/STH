# instale as dependências antes de rodar:
# pip install streamlit sentence-transformers numpy

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ----------------------------
# 1️⃣ Carregar tesauro
# ----------------------------

# Você pode colocar seu sth.txt na mesma pasta ou no diretório do app
TESAURO_FILE = "sth.txt"

# Função para processar o tesauro
def processar_tesauro(file_path):
    termos_principais = []
    sinonimos_por_termo = {}

    with open(file_path, "r", encoding="latin-1") as f:  # latin-1 evita erro de encoding
        linhas = f.readlines()

    termo_atual = None
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        if not linha.startswith(" "):  # termo principal
            termo_atual = linha
            termos_principais.append(termo_atual)
            sinonimos_por_termo[termo_atual] = []
        else:
            if re.match(r"Use: (.+)", linha):
                sinonimo = re.match(r"Use: (.+)", linha).group(1)
                sinonimos_por_termo[termo_atual].append(sinonimo)
            elif re.match(r"Usado por: (.+)", linha):
                conteudo = linha[9:].split("\n")
                for s in conteudo:
                    s = s.strip()
                    if s:
                        sinonimos_por_termo[termo_atual].append(s)
    return termos_principais, sinonimos_por_termo

termos_principais, sinonimos_por_termo = processar_tesauro(TESAURO_FILE)

# ----------------------------
# 2️⃣ Gerar embeddings
# ----------------------------
st.sidebar.info("Carregando modelo, pode demorar alguns segundos na primeira vez...")
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Criar embeddings combinando termo principal + sinônimos
textos_embeddings = [
    termo + " " + " ".join(sinonimos_por_termo[termo])
    for termo in termos_principais
]
embeddings_termos = modelo.encode(textos_embeddings, convert_to_numpy=True)
# Normalizar embeddings
embeddings_termos_norm = embeddings_termos / np.linalg.norm(embeddings_termos, axis=1, keepdims=True)

st.sidebar.success("Modelo e tesauro carregados!")

# ----------------------------
# 3️⃣ Função para sugerir termo principal
# ----------------------------
def sugerir_termo_principal(texto_norma, top_k=1):
    emb_norma = modelo.encode([texto_norma], convert_to_numpy=True)[0]
    emb_norma_norm = emb_norma / np.linalg.norm(emb_norma)
    scores = np.dot(embeddings_termos_norm, emb_norma_norm)
    indices = np.argsort(scores)[::-1][:top_k]
    return [(termos_principais[i], round(scores[i], 3)) for i in indices]

# ----------------------------
# 4️⃣ Interface Streamlit
# ----------------------------
st.title("Bot de Indexação de Normas")

st.markdown("""
Cole o texto da norma abaixo. O bot irá sugerir os termos principais do tesauro correspondentes.
""")

texto_norma = st.text_area("Cole aqui o texto da norma", height=300)

top_k = st.slider("Quantos termos principais sugerir?", 1, 5, 1)

if st.button("Sugerir termos"):
    if texto_norma.strip() == "":
        st.warning("Cole algum texto da norma antes de sugerir!")
    else:
        termos_sugeridos = sugerir_termo_principal(texto_norma, top_k=top_k)
        st.subheader("Termos principais sugeridos:")
        for termo, score in termos_sugeridos:
            st.write(f"- {termo} (similaridade: {score})")
