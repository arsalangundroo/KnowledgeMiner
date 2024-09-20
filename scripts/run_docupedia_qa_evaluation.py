import pandas as pd
from llama_index.core import Settings
from llama_index.core.response_synthesizers import ResponseMode
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.common.blocks.embeddings.hf_embed_models import create_hf_embed_model
from knowledgeminer.common.blocks.llm import meta_llama3
from knowledgeminer.common.blocks.llm.azure_openai_for_llama_index import create_basic_azure_openai_client
from knowledgeminer.evaluation.ragas_evaluation import run_evaluation_with_ground_truth_dataset, create_evaluation_dataset_with_ground_truth_for_RAGAS
from knowledgeminer.prompts.qa_prompts import get_qa_prompt_for_response_synthesizer
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.query_engine.create_query_engine import create_query_engine_from_retriever
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.faiss_vector_store import FaissLamaIndexClient

def run_on_full_pipeline():
    # knowledge_source_uri_list = ['/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl']

    Settings.llm = create_basic_azure_openai_client()
    Settings.embed_model = create_basic_azure_openai_embedding_client()
    EMBED_DIM = 1536

    # Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #                                                tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # Settings.embed_model = create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5")
    # EMBED_DIM = 384

    faiss_client = FaissLamaIndexClient.load_from_persistent_storage("../out/docupedia_faiss_storage",
                                                                     Settings.embed_model,
                                                                     EMBED_DIM)
    vector_index_on_chunks = faiss_client.get_vector_store_index()

    retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks, "docupedia_retriever", similarity_top_k=5)

    response_synthesizer = get_retrieved_context_response_synthesizer(mode=ResponseMode.COMPACT,
                                                                      structured_answer_filtering=False,
                                                                      qa_prompt=get_qa_prompt_for_response_synthesizer())

    query_engine = create_query_engine_from_retriever(retriever, response_synthesizer)

    ragas_eval_dataset = create_evaluation_dataset_with_ground_truth_for_RAGAS(
        "/Users/gar1syv/Documents/ask_bosch_data/docupedia.json", query_engine)

    metrics_df = run_evaluation_with_ground_truth_dataset(ragas_eval_dataset)

    # metrics_df = run_evaluation_with_ground_truth_dataset(None,eval_dataset_json_file="../out/ragas_eval_dataset.json")

    print_relevant_metrics(metrics_df)
    metrics_df.to_csv("../out/ragas_eval_results_for_gpt_k10.csv")


def run_with_existing_inference_results_file(eval_dataset_file):

    metrics_df = run_evaluation_with_ground_truth_dataset(None, eval_dataset_file)

    # metrics_df = run_evaluation_with_ground_truth_dataset(None,eval_dataset_json_file="../out/ragas_eval_dataset.json")

    print(metrics_df.head())
    metrics_df.to_csv("../out/ragas_eval_results_for_gpt.csv")
    print_relevant_metrics(metrics_df)


def print_relevant_metrics(metrics_df,saved_metrics_csv_file=None):
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
    #run_on_full_pipeline()
    run_with_existing_inference_results_file("../out/ragas_eval_dataset.json")
    #print_relevant_metrics(None,"../out/ragas_eval_results_for_gpt.csv")

