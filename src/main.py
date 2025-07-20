from dotenv import load_dotenv
from langchain.globals import set_debug

from src.service.ConectandoComModelos import exemplo_com_gemini_langchain
from src.service.LCELCadeia import usando_cadeia_de_regras_com_obj
from src.service.LCELChatHistory import usando_chat_history

if __name__ == "__main__":
    set_debug(True)
    load_dotenv()
    usando_chat_history()


