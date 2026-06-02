from typing import List

class RetrievalService:

    def __init__(self, retriever):
        self.retriever = retriever   
        
    async def retrieve(self, query: str, top_k: int = 5):
        # # retrieve candidate pool
        # docs = await self.retriever.search(
        #     query,
        #     top_k=20
        # )

        # # rerank
        # reranked = await self.reranker.rerank(
        #     query,
        #     docs
        # )

        # return reranked[:top_k]
        pass