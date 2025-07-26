from dotenv import load_dotenv
from langchain.globals import set_debug

from src.service.RAGservice import pesquisa_com_documento_txt

if __name__ == "__main__":
    set_debug(False)
    load_dotenv()
    pesquisa_com_documento_txt()


