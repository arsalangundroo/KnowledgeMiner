from llama_index.core.retrievers import RecursiveRetriever


def createRecursiveRetrieverFromIndex(vector_store_index, retriever_name, similarity_top_k=3):
    all_nodes = vector_store_index.docstore.docs.values()
    all_nodes_dict = {n.node_id: n for n in all_nodes}
    print(len(all_nodes))
    vector_retriever = vector_store_index.as_retriever(similarity_top_k=similarity_top_k)
    recursive_retriever = RecursiveRetriever(
        retriever_name,
        retriever_dict={retriever_name: vector_retriever},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    return recursive_retriever
