from llama_index.core.query_engine import RetrieverQueryEngine


def create_query_engine_from_retriever(retriever, response_synthesizer,reranker=None):
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer=response_synthesizer,
    )
    return query_engine

