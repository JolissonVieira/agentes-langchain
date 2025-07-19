from openai import OpenAI
class OpenAIClient:
    """Cliente para o modelo OpenAI GPT."""

    def __init__(self, api_key: str, **config):
        self.api_key = api_key
        self.config = config

    def get_model(self, **options) -> OpenAI:
        return OpenAI(
            api_key=self.api_key
        )
