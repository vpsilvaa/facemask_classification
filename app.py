import streamlit as st
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from scipy.spatial.distance import cosine
from numpy.linalg import norm  # Normalização dos embeddings

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializando MTCNN e o modelo de reconhecimento facial
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20, post_process=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Caminho do banco de dados
DATABASE_PATH = "database.npz"

# Carregar banco de dados salvo e normalizar os embeddings
if os.path.exists(DATABASE_PATH):
    database = np.load(DATABASE_PATH, allow_pickle=True)
    database = {name: emb / norm(emb) for name, emb in database.items()}  # Normalizando ao carregar
else:
    database = {}

# Função para obter embedding e normalizá-lo
def get_embedding(image):
    if isinstance(image, np.ndarray):  # Pode acontecer no Streamlit
        image = Image.fromarray(image)
    else:
        image = Image.open(image).convert('RGB')

    face = mtcnn(image)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = model(face).detach().cpu().numpy()[0]
        return embedding / norm(embedding)  # Normalizar antes de salvar ou comparar
    return None

# Função para remover uma pessoa do banco de dados
def remove_from_database(name):
    if name in database:
        del database[name]  # Remove a chave do dicionário
        np.savez_compressed(DATABASE_PATH, **database)  # Atualiza o banco de dados
        return True
    return False

# Interface Streamlit
st.title("Reconhecimento Facial 🔍")

# Escolher entre adicionar, classificar ou remover
option = st.radio("O que deseja fazer?", ["Adicionar nova imagem ao banco", "Classificar uma imagem", "Remover uma pessoa do banco"])

# 📌 Adicionar imagem ao banco
if option == "Adicionar nova imagem ao banco":
    uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])
    name = st.text_input("Digite o nome da pessoa:")

    if st.button("Adicionar"):
        embedding = get_embedding(uploaded_file)
        if embedding is not None:
            database[name] = embedding
            np.savez_compressed(DATABASE_PATH, **database)  # Salvar normalizado
            st.success(f"Imagem de **{name}** adicionada ao banco!")
        else:
            st.error("Nenhum rosto detectado na imagem. Tente outra.")

# 📌 Classificar uma imagem
if option == "Classificar uma imagem":
    uploaded_file = st.file_uploader("Envie uma imagem para classificar", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        query_embedding = get_embedding(uploaded_file)

        if query_embedding is not None:
            st.image(uploaded_file, caption="Imagem enviada", use_column_width=True)

            # Calcular similaridade com embeddings normalizados
            scores = [(cosine(query_embedding, db_emb), name) for name, db_emb in database.items()]
            top_3 = sorted(scores, key=lambda x: x[0])[:3]

            st.subheader("Top 3 possíveis correspondências:")
            for i, (score, name) in enumerate(top_3, 1):
                st.write(f"**{i}. {name}** - Score: {score:.4f}")
        else:
            st.error("Nenhum rosto detectado na imagem. Tente outra.")

# 📌 Remover uma pessoa do banco
if option == "Remover uma pessoa do banco":
    name_to_remove = st.text_input("Digite o nome da pessoa a ser removida:")
    
    if st.button("Remover"):
        if remove_from_database(name_to_remove):
            st.success(f"**{name_to_remove}** foi removido do banco de dados.")
        else:
            st.warning(f"Nome **{name_to_remove}** não encontrado no banco.")

