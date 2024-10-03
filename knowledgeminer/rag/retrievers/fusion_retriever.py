from llama_index.core.retrievers import QueryFusionRetriever


def get_fusion_retriever_with_bm25(vector_retriever, bm25_retriever,num_queries=1,top_k=10):
    #TODO: drop duplicates across two retrievers
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=num_queries,  # set this to 1 to disable query generation
        #mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )

    return retriever