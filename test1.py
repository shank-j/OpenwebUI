"""
title: Llama Index Custom RAG Pipeline
author: open-webui
date: 2024-05-30
version: 1.1
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library, with user-editable valves.
requirements: llama-index, pydantic
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from schemas import OpenAIChatMessage
from llama_index.core.readers.file import PyMuPDFReader

class Pipeline:
    # 1. Define your valves here
    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_MODEL: str = "gpt-3.5-turbo"
        OPENAI_API_TEMPERATURE: float = 0.7
        SYSTEM_PROMPT: str = (
            "You are a knowledgeable assistant who uses Llama Index to answer questions "
            "based on loaded documents."
        )

    def __init__(self):
        # 2. Instantiate valves; OpenWebUI will parse and display them automatically
        self.valves = self.Valves()

        self.documents = None
        self.index = None

    async def on_startup(self):
        import os

        # 3. Use the valves in your startup (e.g., environment variables)
        os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        # If you need to set a custom base URL for OpenAI:
        # os.environ["OPENAI_API_BASE_URL"] = self.valves.OPENAI_API_BASE_URL

        from llama_index.core import VectorStoreIndex

        # Load documents with PyMuPDFReader
        loader = PyMuPDFReader()
        self.documents = loader.load(file_path="sample-new-fidelity-acnt-stmt.pdf")

        # Build the index
        self.index = VectorStoreIndex.from_documents(self.documents)
        print("Llama Index pipeline startup completed.")

    async def on_shutdown(self):
        # Clean up if needed
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Custom RAG pipeline using Llama Index. 
        Retrieves relevant info from your knowledge base and synthesizes a response.
        """
        print("Incoming messages:", messages)
        print("User message:", user_message)

        # 4. Use your valves in the pipeline
        from llama_index import LLMPredictor
        from langchain.chat_models import ChatOpenAI

        # Create an LLM predictor using your valves
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                openai_api_key=self.valves.OPENAI_API_KEY,
                model_name=self.valves.OPENAI_API_MODEL,
                temperature=self.valves.OPENAI_API_TEMPERATURE
            )
        )

        # If you want to add your system prompt as context to the user query
        final_query = f"{self.valves.SYSTEM_PROMPT}\n\nUser: {user_message}"

        # Build a streaming query engine with the custom LLM predictor
        query_engine = self.index.as_query_engine(
            streaming=True,
            llm_predictor=llm_predictor
        )

        # Query the knowledge base
        response = query_engine.query(final_query)

        # Return a streaming response generator (or just `response.response` for a single string)
        return response.response_gen
