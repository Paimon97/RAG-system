class HealthService:

    def __init__(self, retriever, cache):
        self.retriever = retriever
        self.cache = cache

    async def get_health(self):

        try:
            collection_info = (await self.retriever.get_collection_info())
            return {
                "status": "healthy",
                "qdrant": collection_info.get("qdrant_points", 0),
                "elasticsearch": collection_info.get("elasticsearch_docs", 0),
                "cache": ("connected" if self.cache.redis else "disconnected")
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
        

        
        # try:
    #     collection_info = await retriever.get_collection_info()
    #     return {
    #         "status": "healthy",
    #         "qdrant": collection_info.get("qdrant_points", 0),
    #         "elasticsearch": collection_info.get("elasticsearch_docs", 0),
    #         "cache": "connected" if cache.redis else "disconnected"
    #     }
    # except Exception as e:
    #     return {
    #         "status": "error",
    #         "message": str(e)
    #     }