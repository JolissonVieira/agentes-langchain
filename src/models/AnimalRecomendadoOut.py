from pydantic import BaseModel, Field


class AnimalRecomendadoOut(BaseModel):
    melhor_animal: str = Field("Animal que mais se enquadra no meu perfil")
    animais_recomendados: list[str] = Field("Possibilidades de animais que posso escolher")
    animais_nao_recomendados: list[str] = Field("Animais que não se enquadram no meu perfil")


class AnimalAlimentacaoOut(BaseModel):
    comida: str = Field("Qual comida posso alimentar o animal escolhido")
    bebida: str = Field("Qual bebida posso alimentar o animal escolhido")
    forma_alimentacao: str = Field("Qual padrão de alimentação devo seguir e cuidados que devo ter")

class AnimalBrincadeiraOut(BaseModel):
    brincadeiras: list[str] = Field("Quais brincadeiras posso fazer com o animal escolhido")
