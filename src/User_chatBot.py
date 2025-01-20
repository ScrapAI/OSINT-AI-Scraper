#!/usr/bin/python
# -*- coding: utf-8 -*-

import ollama
import logging
import os
from shortterm_memory.ChatbotMemory import ChatbotMemory
from rag.Rag import Rag
from fc.func import get_weather

functions_registry = {
    "get_weather": {
        "func": get_weather,
        "description": "Retourne la météo pour une ville donnée.",
        "parameters": {
            "city": "Nom de la ville à rechercher."
        }
    },
}

def detect_function(input_text: str):
    if "météo" in input_text.lower():
        return "get_weather", {"city": input_text.split()[-1]} 
    return None, {}

class User_chatBot:
    def __init__(self, model:str="mistral:latest", ollama_options=None):
        self.model = model
        self._ollama_option = ollama_options if ollama_options else {'temperature': 1} # vérifier le paramétrage de la température 0 pour déterministe langchain et 1 pour ollama_python
        self.memory = ChatbotMemory()
        self.running = False
        self.response = ""
        self.suma_on_run = False
        self.rag = Rag()
    
    def ans(self, input: str):
        prompt: str
        context: str
        self.running = True
        self.output = ""

        func_name, params = detect_function(input)
        if func_name:
            try:
                func = functions_registry[func_name]['func']
                result = func(**params)
                input += (f" \n voici les resultas fournit par l'outil {func_name} : "+result)
          
            except Exception as e:
                self.output = f"Erreur lors de l'appel de la fonction {func_name} : {str(e)}"
                yield self.output
            return 0

        try:
            mem = self.memory.get_memory()
        except Exception as e:
            self.output = f"Erreur lors de l'appel de la fonction Memoire : {str(e)}"
            mem = ""

        try:
            context = self.rag.search(input)
        except Exception as e:
            print(f"Erreur dans la recherche RAG : {e}")
            context = ""

        prompt = (
            "Vous êtes un assistant intelligent et concis. "
            "Répondez de manière naturelle et directe à la question de l'utilisateur.\n\n"
            "Contexte :\n"
            f"{context}\n\n"
            "Historique de la conversation (à ne pas mentionner directement) :\n"
            f"{mem}\n\n"
            "Question :\n"
            f"{input}\n\n"
            "Réponse :"
        )
        
        response = self.ollama_model.generate(
            prompts=[prompt],
            stream=True
        )
        
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == 'generations':
                generation_list = chunk[1]
                if generation_list and isinstance(generation_list[0], list):
                    generation_chunk = generation_list[0][0] 
                    if hasattr(generation_chunk, 'text'):
                        self.output += generation_chunk.text
                        yield generation_chunk.text
            
        


        
        self.history.save_context({"input": input}, {"output": self.output})
        self.running = False
        return 0
    
if __name__ == "__main__":
    model = User_chatBot(model="mistral:latest")
    model.ans("Hello")
