import streamlit as st
import numpy as np
import re
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

# ----------------------------
# Carregar tesauro
# ----------------------------
TESAURO_FILE = "sth.txt"

def processar_tesauro(file_path):
    termos_principais = []
    sinonimos_por_termo = {}

    with open(file_path, "r", encoding="latin-1") as f:
        linhas = f.readlines()

    termo_atual = None
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        if not linha.startswith(" "):
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
# Modelo de embeddings leve com TF
# ----------------------------
st.sidebar.info("Carregando modelo, aguarde alguns segundos...")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    # Média dos tokens para embedding
    emb = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return tf.math.l2_normalize(emb, axis=1).numpy()[0]

# Pré-computar embeddings dos termos
textos_embeddings = [
    termo + " " + " ".join(sinonimos_por_termo[termo])
    for termo in termos_principais
]
embeddings_termos = np.array([embed_text(t) for t in textos_embeddings])

st.sidebar.success("Modelo e tesauro carregados!")

# ----------------------------
# Função para sugerir termo principal
# ----------------------------
def sugerir_termo_principal(texto_norma, top_k=1):
    emb_norma = embed_text(texto_norma)
    scores = embeddings_termos @ emb_norma
    indices = np.argsort(scores)[::-1][:top_k]
    return [(termos_principais[i], round(scores[i], 3)) for i in indices]

# ----------------------------
# Interface Streamlit
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
