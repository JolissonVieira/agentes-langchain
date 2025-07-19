import google.generativeai as genai
from google.generativeai import GenerativeModel

class GoogleGeminiClient:
    """Cliente para o modelo Google Gemini."""

    def __init__(self, api_key: str, default_options: dict = None, **config):
        if not api_key:
            raise ValueError("A chave da API (api_key) deve ser fornecida.")
        self.api_key = api_key
        self.default_options = default_options or {}
        self.config = config

    def get_model(self, **overrides) -> GenerativeModel:
        """
        Retorna uma instância configurada do modelo Google Gemini.

        :param overrides: Configurações adicionais que sobrescrevem default_options.
        :return: Instância do modelo GenerativeModel.
        """
        genai.configure(api_key=self.api_key)
        options = {**self.default_options, **overrides}
        return genai.GenerativeModel(**options)


