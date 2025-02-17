"""
title: RAG Document Chat Pipeline
author: BrainDrive.ai
date: 2025-02-17
version: 1.0
license: MIT
description: A pipeline to ingest documents, retrieve context from Chroma, and answer questions via OpenWebUI tool calling.
requirements: langchain, langchain-openai, chromadb, pydantic, langchain-community, langchain-text-splitters
"""

import os
from typing import List, Sequence
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


class IngestDocumentInput(BaseModel):
    file_path: str = Field(description="Path to the document (PDF or TXT) to ingest.")


@tool("ingest_document", args_schema=IngestDocumentInput, return_direct=True)
def ingest_document(file_path: str) -> str:
    """Loads a document, splits it into chunks, and stores embeddings in Chroma."""
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vectorstore.add_documents(chunks)
        vectorstore.persist()
        return f"Ingested {len(chunks)} chunks from {file_path}."
    except Exception as e:
        return f"Failed to ingest document: {str(e)}"


class RetrieveAndAnswerInput(BaseModel):
    question: str = Field(description="User's question to retrieve relevant context and generate an answer.")


@tool("retrieve_and_answer", args_schema=RetrieveAndAnswerInput, return_direct=True)
def retrieve_and_answer(question: str) -> str:
    """Retrieves relevant context from the vector database and generates an answer."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant using retrieved context to answer questions."),
            ("user", f"Context:\n{context}\n\nQuestion: {question}")
        ])

        response = model.invoke(prompt.format_prompt())
        return response.content

    except Exception as e:
        return f"Failed to retrieve or answer: {str(e)}"


class Pipeline:
    def __init__(self):
        self.name = "RAG Document Chat"
        self.tools = [ingest_document, retrieve_and_answer]


# Initialize vectorstore outside tools so it's shared
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )
)
