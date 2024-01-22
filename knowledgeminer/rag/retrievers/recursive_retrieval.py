
from llama_index.retrievers import RecursiveRetriever

def createRecursiveRetrieverOnIndex(vectorStoreIndex, all_nodes_dict, similarity_top_k):
    vector_retriever = vectorStoreIndex.as_retriever(similarity_top_k=similarity_top_k)

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    return recursive_retriever
