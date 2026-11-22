from langchain_openai import OpenAIEmbeddings

class EmbeddingService:

    def __init__(
        self,
        model="text-embedding-3-small",
    ):
        self.embeddings = OpenAIEmbeddings(
            model=model
        )

    def embed_docs(
        self,
        texts: list[str]
    ):
        return self.embeddings.embed_documents(
            texts
        )

    def embed_query(
        self,
        text: str
    ):
        return self.embeddings.embed_query(
            text
        )
    