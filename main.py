import streamlit as st
from helper_utils import word_wrap, project_embeddings
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import langchain 
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


st.title("""KI Rechtsanwaltsassistent 
         von Anvin und Sebastian""")

#Client: connecting to OPENAI

openai_key = os.getenv("openai_key")
client = OpenAI(api_key=openai_key)


# Load the data

import streamlit as st
from PyPDF2 import PdfReader  # Importiere den PDF-Reader

uploaded_files = st.file_uploader("Choose files", type=["pdf"], accept_multiple_files=True)

pdf_texts = []

if uploaded_files:  # Prüfen, ob Dateien hochgeladen wurden
    for uploaded_file in uploaded_files:
       # st.write(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Prüfen, ob es sich um eine PDF-Datei handelt
        if uploaded_file.type == "application/pdf":
          #  st.write(f"You uploaded the PDF file: {uploaded_file.name}")

            # PDF-Inhalte extrahieren
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:  # Nur Text hinzufügen, wenn er existiert
                    pdf_texts.append(f"File: {uploaded_file.name}, Page {page_num}:\n{text.strip()}")

    # Nur nicht-leere Texte behalten
    pdf_texts = [text for text in pdf_texts if text]

    # Ausgabe der extrahierten Texte
   # if pdf_texts:
     #   st.write("Extracted PDF text:")
        #for text in pdf_texts:
        #    st.write(text)
   # else:
      #  st.write("No text extracted from the PDFs.")


# Load the embeddings

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


# Split the text into chunks of 1000 characters

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))





# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks: {len(character_split_texts)}")

# Split the text into tokens

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


# initialize the SentenceTransformerEmbeddingFunction in chromadb and create a collection called "microsoft-collection" and add the token_split_texts to the collection





embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()

# Generieren eines eindeutigen Namens
unique_collection_name = f"collection_{uuid.uuid4().hex[:8]}"

# Sammlung erstellen
chroma_collection = chroma_client.create_collection(
    unique_collection_name, embedding_function=embedding_function
)



# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


# Generate hallucinated queries with gpt-3.5-turbo



def generate_multi_query(query, model="gpt-3.5-turbo"):

    prompt = """
Du bist ein sachkundiger Rechtsanwaltsassistent für die Recherche.
Deine Nutzer stellen viele Anfragen zu Briefen und Gerichtsakten.
Für die gegebene Frage schlage bis zu 5 verwandte Fragen vor, um ihnen zu helfen, die benötigten Informationen zu finden.
Stelle prägnante, themenbezogene Fragen (ohne zusammengesetzte Sätze), die verschiedene Aspekte des Themas abdecken.
Stelle sicher, dass jede Frage vollständig und direkt mit der ursprünglichen Anfrage verbunden ist.
Liste jede Frage in einer eigenen Zeile ohne Nummerierung.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

# calling the function to generate hallucinated queries
original_query = st.text_input('Welche Frage zu Fällen soll ich beantworten?')
aug_queries = generate_multi_query(original_query)

# join the original query with the augmented queries

joint_query = [
    original_query
] + aug_queries 


# retrieve the documents from the collection based on joint_query

results = chroma_collection.query(
query_texts=joint_query, n_results=10, include=["documents", "embeddings"])
retrieved_documents = results["documents"]


# get the unique documents

unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


# generate the response from gpt 3.5 turbo based on retrieved documents


def generate_response(original_query, unique_documents):
    context = "\n\n".join(unique_documents)
    prompt = (
        "Du bist ein Assistent für Frage-Antwort-Aufgaben."
        "Nutze die folgenden abgerufenen Kontexte, um die Frage zu beantworten."
        "Verwende nur den abgerufenen Inhalt, verwende nicht dein vorab trainiertes Wissen."
        " Wenn du die Antwort nicht weißt oder die Antwort nicht im gegebenen oder abgerufenen Inhalt enthalten ist, sage, dass du es nicht weißt."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + original_query
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": original_query,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# calling the function to generate the response

answer = generate_response(original_query, unique_documents).content
st.write("Antwort:", answer)