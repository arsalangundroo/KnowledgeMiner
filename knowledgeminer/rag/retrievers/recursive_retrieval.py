from llama_index.retrievers import RecursiveRetriever


def createRecursiveRetrieverOnIndex(vector_store_index, all_nodes_dict, similarity_top_k=3):
    vector_retriever = vector_store_index.as_retriever(similarity_top_k=similarity_top_k, service_context=vector_store_index.service_context)

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    return recursive_retriever
