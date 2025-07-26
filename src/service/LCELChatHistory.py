import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import SecretStr

from src.factory.LLMFactory import LLMFactory
from src.models.AnimalRecomendadoOut import AnimalRecomendadoOut, AnimalAlimentacaoOut, AnimalBrincadeiraOut

memoria = {}
def historico_por_sessao(sessao:str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

def usando_chat_history():

    parser_animal = JsonOutputParser(pydantic_object=AnimalRecomendadoOut)
    parser_alimentacao = JsonOutputParser(pydantic_object=AnimalAlimentacaoOut)
    parser_brincadeira = JsonOutputParser(pydantic_object=AnimalBrincadeiraOut)

    llm = LLMFactory.create_client_langchain(
        model_type="gemini",
        api_key=SecretStr(os.getenv("GEMINI_API_KEY")),
        model="gemini-2.5-flash",
        temperature=0.5
    )

    prompt_animais = PromptTemplate(
        template="""
            Sugira qual animal de estimação eu poderia adotar de acordo com os meus dados.se atentando tanbém aos que não posso.{dados}
        {formato_saida}
        """,
        input_variables=["dados"],
        partial_variables={"formato_saida": parser_animal.get_format_instructions()}
    )

    prompt_alimentacao = PromptTemplate(
        template="""
                Sugira qual melhor forma de aliemtar o animal escolhido.{dados}
            {formato_saida}
            """,
        input_variables=["dados"],
        partial_variables={"formato_saida": parser_alimentacao.get_format_instructions()}
    )

    prompt_brincadeira = PromptTemplate(
        template="""
                    Sugira qual melhores brincadeiras posso fazer com o animal escolhido{dados}
                {formato_saida}
                """,
        input_variables=["dados"],
        partial_variables={"formato_saida": parser_brincadeira.get_format_instructions()}
    )
    prompt_sugestao = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(prompt=prompt_animais),
        HumanMessagePromptTemplate(prompt=prompt_alimentacao),
        HumanMessagePromptTemplate(prompt=prompt_brincadeira),
        MessagesPlaceholder(variable_name="historico")
    ])

    cadeia = prompt_sugestao | llm | StrOutputParser()


    cadeia_completa_com_memoria = RunnableWithMessageHistory(
        runnable=cadeia,
        get_session_history=historico_por_sessao,
        input_messages_key="dados",
        history_messages_key="historico"
    )

    dados_entrada = """
           tipo_casa: apartamento
           tamanho_espaço: pequeno
           tempo_disponivel: pouco
           alergia: pelos
       """

    resultado = cadeia_completa_com_memoria.invoke(
        {
            "dados":dados_entrada
        },
        config={"session_id": "estudo_curso_alura"}
    )

    print(resultado)



