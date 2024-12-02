import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile

# Configuration de l'application Streamlit
st.title("Explorez Vos Documents Avec Intelligence")

# Choix du modèle LLM avec les modèles disponibles
model_choices = {
    "glm4:latest": {"dimension": 4096},  # Adapté à votre modèle glm4
    "nomic-embed-text:latest": {"dimension": 274},  # Adapté à nomic-embed-text
    "qwen2.5-coder:latest": {"dimension": 3584},
    "llama3.1:8b": {"dimension": 4096},
    "llama2:latest": {"dimension": 3072},
}

selected_model = st.selectbox("Sélectionnez le modèle que vous souhaitez utiliser :", list(model_choices.keys()))

# Récupération des paramètres du modèle sélectionné
model_info = model_choices[selected_model]
embedding_dim = model_info["dimension"]

# Nettoyer le nom du modèle pour qu'il soit valide comme répertoire
valid_model_name = selected_model.replace(":", "_")

# Spécifier le répertoire pour la base de données vectorielle
persist_directory = f"./chroma_db_{valid_model_name}"

# Configuration du modèle d'embedding et du LLM
embed_model = OllamaEmbeddings(model=selected_model, base_url="http://127.0.0.1:11434")
llm = Ollama(model=selected_model, base_url="http://127.0.0.1:11434")

# Vérifiez si une base de données existe déjà pour ce modèle, sinon créez-en une
if not os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
else:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# Création d'un récupérateur (Retriever)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Chargement d'un prompt prédéfini pour le chat de questions-réponses
retrievel_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Création de la chaîne de combinaison de documents
combine_docs_chain = create_stuff_documents_chain(llm, retrievel_qa_chat_prompt)

# Création de la chaîne de récupération finale
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Section pour télécharger un fichier PDF
uploaded_file = st.file_uploader("Téléchargez un document PDF", type=["pdf"])

if uploaded_file is not None:
    # Enregistrer le fichier téléchargé dans un emplacement temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Extraction du texte du document PDF
    with st.spinner("Extraction du texte du PDF..."):
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Découper le texte en segments
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Ajouter les documents à la base de données vectorielle
        vector_store.add_documents(texts)

        st.success("Texte extrait et ajouté à la base de données.")

    # Champ de saisie pour la question de l'utilisateur
    question = st.text_input("Posez votre question ici :")

    # Bouton pour soumettre la question
    if st.button("Obtenir la réponse"):
        if question:
            with st.spinner("Génération de la réponse..."):
                response = retrieval_chain.invoke({"input": question})
                st.write("Réponse :", response["answer"])
        else:
            st.write("Veuillez entrer une question.")

