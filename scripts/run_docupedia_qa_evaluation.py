from llama_index.core import Settings
from llama_index.core.response_synthesizers import ResponseMode
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.common.blocks.llm.azure_openai_for_llama_index import create_basic_azure_openai_client
from knowledgeminer.evaluation.ragas_evaluation import run_evaluation_with_ground_truth_dataset, create_evaluation_dataset_with_ground_truth_for_RAGAS
from knowledgeminer.prompts.qa_prompts import get_qa_prompt_for_response_synthesizer
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.query_engine.create_query_engine import create_query_engine_from_retriever
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.faiss_vector_store import FaissLamaIndexClient

if __name__ == "__main__":

    # docupedia_docs = load_html_content_from_jsonl_field("/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl")
    # print(len(docupedia_docs))
    # print(docupedia_docs[0])

    knowledge_source_uri_list = ['/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl']

    Settings.llm = create_basic_azure_openai_client()
    Settings.embed_model = create_basic_azure_openai_embedding_client()
    EMBED_DIM = 1536

    # Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #                                                tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # Settings.embed_model = create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5")
    # EMBED_DIM = 384

    faiss_client = FaissLamaIndexClient.load_from_persistent_storage("../out/docupedia_faiss_storage", Settings.embed_model,
                                                                     EMBED_DIM)
    vector_index_on_chunks = faiss_client.get_vector_store_index()

    retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks, "docupedia_retriever", similarity_top_k=3)

    response_synthesizer = get_retrieved_context_response_synthesizer(mode=ResponseMode.COMPACT, structured_answer_filtering=False, qa_prompt=get_qa_prompt_for_response_synthesizer())

    query_engine = create_query_engine_from_retriever(retriever, response_synthesizer)

    ragas_eval_dataset = create_evaluation_dataset_with_ground_truth_for_RAGAS("/Users/gar1syv/Documents/ask_bosch_data/docupedia.json",query_engine)

    metrics_df = run_evaluation_with_ground_truth_dataset(ragas_eval_dataset)

    #metrics_df = run_evaluation_with_ground_truth_dataset(None,eval_dataset_json_file="../out/ragas_eval_dataset.json")

    print(metrics_df.head())
    metrics_df.to_csv("../out/ragas_eval_results.csv")

