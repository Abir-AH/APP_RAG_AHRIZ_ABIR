## Système d'Interrogation de Documents avec LLM et Récupération
Une application Streamlit permettant d'explorer des documents PDF à l'aide d'intelligence artificielle. L'application utilise des modèles de langage pour répondre aux questions sur le contenu des documents téléchargés. L'application supporte plusieurs langues (Français, Anglais, Arabe) et permet à l'utilisateur de poser des questions sur les documents qu'il télécharge.

## Fonctionnalités

- **Téléchargement de documents PDF** : Vous pouvez télécharger vos documents PDF pour les analyser.
- **Sélection du modèle de langage** : Choisissez parmi plusieurs modèles de langage pour l'extraction et l'interrogation des documents.
- **Langue multiple** : L'application prend en charge trois langues : Français, Anglais et Arabe.
- **Réponses à des questions** : Vous pouvez poser des questions sur les documents téléchargés et obtenir des réponses générées par l'intelligence artificielle basée sur le contenu du document.
- **Traitement en arrière-plan** : L'application utilise un thread d'exécution pour extraire et analyser les documents en arrière-plan tout en vous permettant d'interagir avec l'interface.

## Prérequis

Avant de pouvoir utiliser l'application, vous devez avoir installé les bibliothèques suivantes :

- `streamlit`
- `langchain`
- `langchain_community`
- `PyPDF2`
- `langdetect`
- `chromadb`

Vous pouvez installer ces bibliothèques avec la commande suivante :


pip install -r requirements.txt

### Installation

## Prérequis

Assurez-vous d'avoir Python 3.7+ installé sur votre machine. Vous pouvez vérifier la version avec la commande suivante :

python --version

## Utilisation
# Lancer l'application
1- Clonez ce repository ou téléchargez les fichiers sources.
2- Installez les dépendances requises en utilisant pip install -r requirements.txt.
3- Lancez l'application Streamlit :
streamlit run app.py
4- Ouvrez l'URL générée dans votre navigateur pour interagir avec l'application.
# Étapes pour utiliser l'application

1-  Sélectionnez la langue : Choisissez la langue d'interface (Français, Anglais, ou Arabe).
2- Téléchargez un document PDF : Cliquez sur le bouton pour télécharger un document PDF à analyser.
3- Sélectionnez le modèle LLM : Choisissez le modèle de langage que vous souhaitez utiliser pour l'analyse du document.
4- Posez une question : Entrez une question liée au contenu du document téléchargé.
5- Recevez la réponse : L'IA génère une réponse basée sur le contenu du document.
# Code Explicationérée dans votre navigateur pour interagir avec l'application.
- 'Langchain : Utilisé pour la gestion des embeddings et des modèles de langage (LLM). Le modèle sélectionné est utilisé pour générer des embeddings et interagir avec le document.'
- 'Chroma : Base de données vectorielle pour stocker et rechercher des documents extraits.'
- 'Ollama : Utilisé pour l'embedding et l'inférence via des modèles pré-entrainés.'
- 'ThreadPoolExecutor : Permet d'exécuter l'extraction des documents en arrière-plan pendant que l'utilisateur pose sa question.'
- 'Langdetect : Détecte la langue de la question pour générer la réponse dans la langue appropriée.'
# Contribuer
Si vous souhaitez contribuer à ce projet, voici comment vous pouvez participer :

1- Fork ce repository.

2- Créez une branche pour votre fonctionnalité (git checkout -b feature-nouvelle-fonctionnalite).
3- Faites vos modifications et ajoutez des tests si nécessaire.
4- Commitez vos modifications (git commit -am 'Ajout d'une nouvelle fonctionnalité').
5- Poussez votre branche (git push origin feature-nouvelle-fonctionnalite).
6- Créez une Pull Request.
#Auteurs
Développé par AHRIZ ABIR.
## Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

### Explication des sections du `README.md` :

- **Introduction** : Présente brièvement l'objectif de l'application.
- **Fonctionnalités** : Détaille les fonctionnalités principales de l'application.
- **Prérequis** : Liste les bibliothèques nécessaires pour exécuter le projet.
- **Utilisation** : Donne des instructions sur comment démarrer l'application.
- **Code Explication** : Explique brièvement le rôle des principales bibliothèques et fonctions utilisées dans le projet.
- **Contribuer** : Fournit des instructions pour contribuer au projet.
- **Auteurs** : Mentionne l'auteur du projet.
- **Licence** : Spécifie la licence utilisée pour le projet (vous pouvez ajuster la licence selon le choix que vous faites).

N'hésitez pas à personnaliser ce `README.md` en fonction des spécificités de votre projet et des besoins des utilisateurs.
