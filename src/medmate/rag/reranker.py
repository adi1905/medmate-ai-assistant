# from sentence_transformers import CrossEncoder
from .hybrid_search import SearchResult
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# class BGERanker:
#     def __init__(
#         self,
#         model_name: str = "BAAI/bge-reranker-v2-m3",
#     ):
#         self.model = CrossEncoder(
#             model_name,
#             trust_remote_code=True,
#         )
#     def rerank(
#         self,
#         query: str,
#         results: list[SearchResult],
#         top_k: int | None = None,
#     ) -> list[SearchResult]:

#         if not results:
#             return []

#         pairs = [
#             (
#                 query,
#                 r.document.page_content,
#             )
#             for r in results
#         ]

#         scores = self.model.predict(
#             pairs,
#             show_progress_bar=False,
#         )

#         ranked = []

#         for result, score in zip(results, scores):
#             ranked.append(
#                 SearchResult(
#                     document=result.document,
#                     score=float(score),
#                     search_type=result.search_type,
#                 )
#             )

#         ranked.sort(
#             key=lambda x: x.score,
#             reverse=True,
#         )

#         if top_k:
#             ranked = ranked[:top_k]

#         return ranked
    


class CohereRanker:
    def __init__(
        self,
        model_name: str = "rerank-v3.5",
    ):
        self.model_name = model_name
        self.client = cohere.ClientV2(api_key=COHERE_API_KEY)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        print("Called cohere reranker")
        if not results:
            return []

        documents = [r.document.page_content for r in results]

        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=top_k,  # None returns all results, ranked
        )

        # Cohere returns {index, relevance_score} pairs, NOT the documents
        # themselves — index refers back into the `documents` list we sent,
        # which is positionally identical to `results`. Map back through it.
        ranked = []
        for item in response.results:
            original = results[item.index]
            ranked.append(
                SearchResult(
                    document=original.document,
                    score=float(item.relevance_score),
                    search_type=original.search_type,
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked