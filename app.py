from concurrent.futures import ThreadPoolExecutor
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
import langdetect  # Module pour la détection de la langue

# Configuration de l'application Streamlit
st.title("Système d'Interrogation de Documents avec LLM et Récupération")

# Choix de la langue
language_choices = ["français", "english", "arabic"]
selected_language = st.selectbox("Sélectionnez la langue", language_choices)

# Traductions pour les différentes langues
translations = {
    "français": {
        "model_selector": "Sélectionnez le modèle que vous souhaitez utiliser :",
        "upload_file": "Téléchargez un document PDF",
        "ask_question": "Posez votre question ici :",
        "loading_documents": "Extraction des documents...",
        "loading_response": "Génération de la réponse...",
        "question_placeholder": "Veuillez poser une question."
    },
    "english": {
        "model_selector": "Select the model you want to use:",
        "upload_file": "Upload a PDF document",
        "ask_question": "Ask your question here:",
        "loading_documents": "Extracting documents...",
        "loading_response": "Generating response...",
        "question_placeholder": "Please ask a question."
    },
    "arabic": {
        "model_selector": "اختر النموذج الذي تريد استخدامه:",
        "upload_file": "قم بتحميل مستند PDF",
        "ask_question": "اطرح سؤالك هنا:",
        "loading_documents": "جاري استخراج المستندات...",
        "loading_response": "جاري إنشاء الإجابة...",
        "question_placeholder": "يرجى طرح سؤال."
    }
}

# Choix du modèle LLM
model_choices = {
    "llama2:latest": {"dimension": 4096},
    "nomic-embed-text:latest": {"dimension": 768},
    "qwen2.5-coder:latest": {"dimension": 5120},
    "glm4:latest": {"dimension": 5120},
}

# Ajout d'une clé unique pour le selectbox afin d'éviter le conflit
selected_model = st.selectbox(translations[selected_language]["model_selector"], list(model_choices.keys()), key="model_selector")

# Récupération des paramètres du modèle sélectionné
model_info = model_choices[selected_model]
embedding_dim = model_info["dimension"]

# Spécifier le répertoire pour la base de données vectorielle (remplacer les caractères invalides)
persist_directory = f"./chroma_db_{selected_model.replace(':', '_')}"

# Configuration du modèle d'embedding et du LLM
embed_model = OllamaEmbeddings(model=selected_model, base_url="http://127.0.0.1:11434")
llm = Ollama(model=selected_model, base_url="http://127.0.0.1:11434")

# Vérification de la base de données existante ou création d'une nouvelle
if not os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
else:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# Création d'un récupérateur (Retriever)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Chargement d'un prompt prédéfini pour le chat de questions-réponses
retrievel_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Création de la chaîne de combinaison de documents
combine_docs_chain = create_stuff_documents_chain(llm, retrievel_qa_chat_prompt)

# Création de la chaîne de récupération finale
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Fonction pour extraire et ajouter les documents en arrière-plan
def extract_and_add_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vector_store.add_documents(texts)
    return "Texte extrait et ajouté à la base de données."

# Fonction pour détecter la langue de la question
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return 'unknown'

# Section pour télécharger un fichier PDF
uploaded_file = st.file_uploader(translations[selected_language]["upload_file"], type=["pdf"])

# Champ de saisie pour la question de l'utilisateur
question = st.text_input(translations[selected_language]["ask_question"])

# Utilisation de ThreadPoolExecutor pour extraire les documents en arrière-plan
with ThreadPoolExecutor() as executor:
    if uploaded_file is not None:
        with st.spinner(translations[selected_language]["loading_documents"]):
            future = executor.submit(extract_and_add_documents, uploaded_file)
            st.success("Extraction en cours en arrière-plan. Posez votre question pendant ce temps.")

    # Répondre à la question sur les documents téléchargés
    if question:
        with st.spinner(translations[selected_language]["loading_response"]):
            response_from_docs = retrieval_chain.invoke({"input": question})
            if not response_from_docs["answer"]:
                response_from_llm = llm.generate([question])
                st.write("Réponse générée par le LLM :", response_from_llm.generations[0][0].text)
            else:
                # Détecter la langue de la question et ajuster la réponse
                lang = detect_language(question)
                
                if lang == 'arabic':
                    st.write("إجابة بناءً على المستندات :", response_from_docs["answer"])  # Réponse en arabe
                elif lang == 'français':
                    st.write("Réponse basée sur les documents :", response_from_docs["answer"])  # Réponse en français
                else:
                    st.write("Based on the documents:", response_from_docs["answer"])  # Réponse en anglais
    else:
        st.write(translations[selected_language]["question_placeholder"])
