from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.postprocessor.colbert_rerank import ColbertRerank


def get_flag_embedding_reranker(model_name="BAAI/bge-reranker-base", top_n=5,use_fp16=False):
    reranker = FlagEmbeddingReranker(model=model_name, top_n=top_n)
    return reranker


def get_colbert_reranker(model_name="colbert-ir/colbertv2.0",tokenizer_name="colbert-ir/colbertv2.0",top_n=5):
    colbert_reranker = ColbertRerank(
        top_n=top_n,
        model=model_name,
        tokenizer=tokenizer_name,
        keep_retrieval_score=True,
    )

    return colbert_reranker