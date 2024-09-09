from llama_index.core.query_engine import RetrieverQueryEngine

def create_query_engine_from_retriever(retriever, response_synthesizer):
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine

