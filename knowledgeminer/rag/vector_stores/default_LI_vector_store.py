from llama_index import VectorStoreIndex, ServiceContext
from llama_index import set_global_service_context

def create_vector_store_index(input_nodes, llm, embedding_model):
    service_context = None
    if llm and embedding_model:
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding_model,
        )

    print("Creating Vector Store Index ............")
    vector_index = VectorStoreIndex(input_nodes, service_context=service_context)
    print("Vector Store Creation Complete!")
    return vector_index
