from typing import List, Dict


class MessageFactory:
    """
    Factory para criar mensagens no formato esperado por LLMs.
    """

    @staticmethod
    def create_message(role: str, content: str) -> Dict[str, str]:
        """
        Cria uma mensagem no formato esperado com base na role solicitada.

        Args:
            role (str): O tipo da mensagem. Pode ser "system", "user" ou "assistant".
            content (str): O conteúdo da mensagem.

        Returns:
            dict: Mensagem formatada no padrão { "role": role, "content": content }.
        """
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"O tipo de mensagem '{role}' não é válido. Use 'system', 'user' ou 'assistant'.")
        return {"role": role, "text": content}

    @staticmethod
    def system(content: str) -> Dict[str, str]:
        """
        Cria uma mensagem do tipo 'system'.

        Args:
            content (str): O conteúdo da mensagem do sistema.

        Returns:
            dict: Mensagem formatada como 'system'.
        """
        return MessageFactory.create_message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Dict[str, str]:
        """
        Cria uma mensagem do tipo 'user'.

        Args:
            content (str): O conteúdo da mensagem enviada pelo usuário.

        Returns:
            dict: Mensagem formatada como 'user'.
        """
        return MessageFactory.create_message(role="user", content=content)

    @staticmethod
    def assistant(content: str) -> Dict[str, str]:
        """
        Cria uma mensagem do tipo 'assistant'.

        Args:
            content (str): O conteúdo da mensagem gerada pelo assistente.

        Returns:
            dict: Mensagem formatada como 'assistant'.
        """
        return MessageFactory.create_message(role="assistant", content=content)


