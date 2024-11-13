import pandas as pd
from llama_index.core import Settings
from llama_index.core.response_synthesizers import ResponseMode
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
from knowledgeminer.rag.retrievers.bm25_retriever import get_bm25_retriever
from knowledgeminer.rag.retrievers.fusion_retriever import get_fusion_retriever_with_bm25
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.common.blocks.embeddings.hf_embed_models import create_hf_embed_model
from knowledgeminer.common.blocks.llm import meta_llama3
from knowledgeminer.common.blocks.llm.azure_openai_for_llama_index import create_basic_azure_openai_client
from knowledgeminer.evaluation.ragas_evaluation import run_evaluation_with_ground_truth_dataset, \
    create_evaluation_dataset_with_ground_truth_for_RAGAS
from knowledgeminer.prompts.qa_prompts import get_qa_prompt_for_response_synthesizer
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.query_engine.create_query_engine import create_query_engine_from_retriever
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.faiss_vector_store import FaissLamaIndexClient
from knowledgeminer.rag.rankers.llama_index_rerankers import get_colbert_reranker, get_flag_embedding_reranker
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform, HyDEQueryTransform,
)
from llama_index.core.query_engine import MultiStepQueryEngine, TransformQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from knowledgeminer.rag.preprocessors.node_processors import DocumentsToNodesProcessor
from knowledgeminer.rag.loaders.load_documents_from_json_fields import load_html_content_from_jsonl_field, \
    load_processed_docupedia_docs_from_jsonl_field


def run_on_full_pipeline():
    # knowledge_source_uri_list = ['/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl']
    EVAL_DATASET_OUTFILE = "../out/ragas_eval_dataset_llama3_k5_multi_e5_reranked_colbert_DET_PROMPT.json"
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    EMBED_DIM = 1024
    # EMBED_DIM = 384 # "BAAI/bge-small-en-v1.5"
    # EMBED_DIM = 768  # "BAAI/bge-base-en-v1.5"
    # EMBED_DIM = 3584  # "BAAI/bge-multilingual-gemma2"

    # Settings.llm = create_basic_azure_openai_client()
    # Settings.embed_model = create_basic_azure_openai_embedding_client()
    # EMBED_DIM = 1536

    # TODO: Use bge-large or bge-base embedding model:
    Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                                                   tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    Settings.embed_model = create_hf_embed_model(model_name=EMBED_MODEL_NAME)

    # print(Settings.embed_model._model.eval())

    faiss_client = FaissLamaIndexClient.load_from_persistent_storage("../out/docupedia_faiss_storage_multi_ling",
                                                                     Settings.embed_model,
                                                                     EMBED_DIM)
    vector_index_on_chunks = faiss_client.get_vector_store_index()

    vector_retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks, "docupedia_retriever",
                                                         similarity_top_k=10)

    retriever = vector_retriever

    # print(f"Docstore length for bm25: {len(vector_index_on_chunks.docstore.docs)}")
    # bm25_retriever = get_bm25_retriever(vector_index_on_chunks.docstore.docs,similarity_top_k=3)

    # retriever = get_fusion_retriever_with_bm25(vector_retriever,bm25_retriever,top_k=10)

    response_synthesizer = get_retrieved_context_response_synthesizer(mode=ResponseMode.COMPACT,
                                                                      structured_answer_filtering=False,
                                                                      qa_prompt=get_qa_prompt_for_response_synthesizer())

    reranker = get_colbert_reranker(top_n=5)
    query_engine = create_query_engine_from_retriever(retriever, response_synthesizer, reranker=reranker)

    ############# Reomve this block later ##########################
    # step_decompose_transform = StepDecomposeQueryTransform(Settings.llm, verbose=True)

    # query_engine = MultiStepQueryEngine(
    #     query_engine, query_transform=step_decompose_transform
    # )

    # hyde = HyDEQueryTransform(include_original=True)
    # query_engine = TransformQueryEngine(base_query_engine, hyde)

    ########################################################

    ragas_eval_dataset = create_evaluation_dataset_with_ground_truth_for_RAGAS(
        "./data/docupedia.json", query_engine, out_file=EVAL_DATASET_OUTFILE)

    metrics_df = run_evaluation_with_ground_truth_dataset(ragas_eval_dataset)

    # metrics_df = run_evaluation_with_ground_truth_dataset(None,eval_dataset_json_file="../out/ragas_eval_dataset.json")

    print_relevant_metrics(metrics_df)

    metrics_df.to_csv("../out/ragas_eval_results_for_llama3_reranked.csv")


def run_with_existing_inference_results_file(eval_dataset_file):
    # Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #                                                tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # Settings.embed_model = create_hf_embed_model(model_name="BAAI/bge-base-en-v1.5")

    metrics_df = run_evaluation_with_ground_truth_dataset(None, eval_dataset_file)

    # metrics_df = run_evaluation_with_ground_truth_dataset(None,eval_dataset_json_file="../out/ragas_eval_dataset.json")

    print(metrics_df.head())
    metrics_df.to_csv("./out/ragas_eval_results_for_llama3.csv")
    print_relevant_metrics(metrics_df)


def print_relevant_metrics(metrics_df, saved_metrics_csv_file=None):
    if metrics_df is None:
        metrics_df = pd.read_csv(saved_metrics_csv_file)

    x = metrics_df.describe()
    columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    for col in columns:
        print(f"'{col}':")
        print(f"Mean: {x[col]['mean']}")
        print(f"St. Deviation: {x[col]['std']}")
        print(f"25%: {x[col]['25%']}")
        print(f"50%: {x[col]['50%']}")
        print("================================")

    mean_retrieval_f1 = (2 * (x["context_precision"]['mean'] * x["context_recall"]['mean'])) / (
            x["context_precision"]['mean'] + x["context_recall"]['mean'])
    print("Mean_Retrieval_F1: " + str(mean_retrieval_f1))


if __name__ == "__main__":
    run_on_full_pipeline()
    # run_with_existing_inference_results_file("../out/ragas_eval_dataset_for_llama3.json")
    # print_relevant_metrics(None,"../out/ragas_eval_results_for_gpt.csv")

