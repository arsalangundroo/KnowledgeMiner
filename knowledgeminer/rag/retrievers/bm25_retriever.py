from llama_index.retrievers.bm25 import BM25Retriever


def get_bm25_retriever(documents, similarity_top_k=2):

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=documents, similarity_top_k=similarity_top_k
    )
    return bm25_retriever