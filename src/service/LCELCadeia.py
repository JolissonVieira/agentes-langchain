import os

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from pydantic import SecretStr

from src.factory.LLMFactory import LLMFactory
from src.models.AnimalRecomendadoOut import AnimalRecomendadoOut, AnimalAlimentacaoOut, AnimalBrincadeiraOut


def extrair_dados_animal(output_animal):
    return {"dados": f"Animal recomendado: {output_animal['melhor_animal']}"}

def usando_cadeia_de_regras_com_obj():

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
                    Sugira qual melhores brincadeiras posso fazer com o animal escolhido.{dados}
                {formato_saida}
                """,
        input_variables=["dados"],
        partial_variables={"formato_saida": parser_brincadeira.get_format_instructions()}
    )

    cadeia_animal = prompt_animais | llm | parser_animal
    adaptador_para_alimentacao = RunnableLambda(extrair_dados_animal)
    cadeia_alimentacao = prompt_alimentacao | llm | parser_alimentacao
    cadeia_brincadeira = prompt_brincadeira | llm | parser_brincadeira

    cadeia_completa = cadeia_animal | adaptador_para_alimentacao | RunnableMap({
        "animal": cadeia_animal,
        "alimentacao": cadeia_alimentacao,
        "brincadeira": cadeia_brincadeira
    })


    dados_entrada = {
        "dados": {
            "tipo_casa": "apartamento",
            "tamanho_espaço": "pequeno",
            "tempo_disponivel": "pouco",
            "alergia": "pelos"
        }
    }

    resultado = cadeia_completa.invoke(dados_entrada)
    print(resultado)



