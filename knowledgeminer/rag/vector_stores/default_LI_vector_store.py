from llama_index import VectorStoreIndex, ServiceContext



def create_vector_store_index(input_nodes, service_context):
    vector_index = VectorStoreIndex(input_nodes, service_context=service_context)
    return vector_index