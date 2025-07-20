import os

from langchain.chains.summarize.map_reduce_prompt import prompt_template
from pydantic import SecretStr

from src.factory.LLMFactory import LLMFactory
from src.factory.MessageFactory import MessageFactory
from langchain.prompts import PromptTemplate

def exemplo_com_openai():
    llm = LLMFactory.create_client(
        model_type="openai",
        api_key=os.getenv("OPENAI_API_KEY")
    ).get_model()

    numero_dias = 7
    numero_criancao = 2
    atividade = "musica"
    prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_criancao} crianças, que goste de {atividade}"

    resposta = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            MessageFactory.system("Você é um assistente de roteiro de viagens"),
            MessageFactory.user(prompt)
        ]
    )

    print(resposta)


def exemplo_com_openai_langchain():
    llm = LLMFactory.create_client_langchain(
        model_type="openai",
        api_key=SecretStr(os.getenv("OPENAI_API_KEY")),
        model="gpt-3.5-turbo",
        temperature=0.5
    )

    numero_dias = 7
    numero_criancao = 2
    atividade = "musica"
    prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_criancao} crianças, que goste de {atividade}"

    resposta = llm.invoke(prompt)

    print(resposta)

def exemplo_com_gemini_langchain():
    llm = LLMFactory.create_client_langchain(
        model_type="gemini",
        api_key=SecretStr(os.getenv("GEMINI_API_KEY")),
        model="gemini-2.5-flash",
        temperature=0.5
    )

    prompt_template = PromptTemplate(
        template="""
        Crie um roteiro de viagem de {numero_dias} dias, 
        para uma família com {numero_criancao} crianças, que goste de {atividade}
        """)

    prompt_final = prompt_template.format(numero_dias=7, numero_criancao=2, atividade="praia")
    resposta = llm.invoke(prompt_final)

    print(resposta.content)