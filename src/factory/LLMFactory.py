from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.client.GoogleGeminiClient import GoogleGeminiClient
from src.client.OpenAIClient import OpenAIClient


class LLMFactory:
    """Factory para criar instâncias de clientes de LLM (OpenAI ou Google Gemini)."""

    @staticmethod
    def create_client(model_type: str, api_key: str, **kwargs):
        """
        Cria uma instância de um cliente de LLM específico com base no tipo de modelo solicitado.

        Args:
            model_type (str): O tipo do modelo. Ex.: 'openai', 'gemini'.
            api_key (str): A chave da API correspondente ao modelo.
            **kwargs: Parâmetros adicionais para o cliente (opcionais).

        Returns:
            Instância do cliente correspondente ao modelo escolhido.
        """
        if model_type == "openai":
            return OpenAIClient(api_key, **kwargs)
        elif model_type == "gemini":
            return GoogleGeminiClient(api_key, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' não é suportado.")

    @staticmethod
    def create_client_langchain(model_type: str, api_key: SecretStr, **kwargs):
        """
        Cria uma instância de um cliente de LLM específico com base no tipo de modelo solicitado.

        Args:
            model_type (str): O tipo do modelo. Ex.: 'openai', 'gemini'.
            api_key (str): A chave da API correspondente ao modelo.
            **kwargs: Parâmetros adicionais para o cliente (opcionais).

        Returns:
            Instância do cliente correspondente ao modelo escolhido.
        """
        if model_type == "openai":
            return ChatOpenAI(api_key=api_key ,**kwargs)
        elif model_type == "gemini":
            return ChatGoogleGenerativeAI(google_api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' não é suportado.")
