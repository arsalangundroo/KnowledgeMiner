from llama_index import VectorStoreIndex, ServiceContext


def create_vector_store_index(input_nodes, llm, embedding_model):
    service_context = None
    if llm and embedding_model:
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding_model,
        )
    vector_index = VectorStoreIndex(input_nodes, service_context=service_context)
    return vector_index
