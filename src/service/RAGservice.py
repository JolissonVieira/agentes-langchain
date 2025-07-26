import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from src.factory.LLMFactory import LLMFactory


def pesquisa_com_documento_txt():

    llm = LLMFactory.create_client_langchain(
        model_type="gemini",
        api_key=SecretStr(os.getenv("GOOGLE_API_KEY")),
        model="gemini-2.5-flash",
        temperature=0.5
    )
    os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    document = TextLoader(
        "documentos/GTB_gold_Nov23.txt",
        encoding="utf-8"
    )

    pedacos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    ).split_documents(document.load())


    dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

    prompt_consulta_seguro  = ChatPromptTemplate.from_messages(
        [
            ("system","Responda usando exclusivamente o conteúdo fornecido."),
            ("human","{query}\n\nContexto: \n{context}\n\nResposta:")
        ]
    )

    cadeia = prompt_consulta_seguro | llm | StrOutputParser()
    pergunta = "Como devo proceder caso tenha um item roubado?"
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    final = cadeia.invoke({"query": pergunta, "context": contexto})
    print(final)