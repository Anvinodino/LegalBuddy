# Chatbot

This repo covers a capstone Project, which builds a chatbot based on RAG.
It lets you upload multiple pdf-documents and then the llm model answers you question regarding the pdf-documents. Its use-case can be in Legal Tech. Lawyeers are working on many cases simultanously, so it is very hard to keep track the details of each case (eg.: Client asks how much is the granted compensation ruled by the court). The Lawyer can ask the client's question the RAG-Model.


# What to find in this repo

In the repo is the requirements.txt - file included, which you have to use in order to run the code in the main.py and rag_prototype.ipynb.
The rag_prototype.ipynb is a notebook which explains all the code in a snippet briefly and the main.py covers all the code combined with a simple userinterface with Streamlit


# Requirements

Installing the virtual environment and the required packages in order to run this code
   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt


Note: You must also create an OPENAI Account with an OPENAI API-Key and a minimum of 5 Dollars as Deposit. Store the key in the .env file. This repo does not include the .env file since you have to create your own with the OPENAI API-Key (acquired here: https://platform.openai.com/settings/organization/api-keys )
