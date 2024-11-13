import os
from typing import List
import json

from pathlib import Path
import sys

from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever

from knowledgeminer.rag.retrievers.bm25_retriever import get_bm25_retriever
from knowledgeminer.rag.retrievers.fusion_retriever import get_fusion_retriever_with_bm25

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.rag.loaders.load_documents_from_json_fields import load_html_content_from_jsonl_field, \
    load_processed_docupedia_docs_from_jsonl_field
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.preprocessors.node_processors import DocumentsToNodesProcessor
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.chroma_db import ChromaDBLamaIndexClient
from llama_index.core.response_synthesizers import ResponseMode
from knowledgeminer.common.blocks.llm import azure_openai_for_llama_index, meta_llama3
from knowledgeminer.common.blocks.embeddings.hf_embed_models import create_llama_index_hf_embed_model
from knowledgeminer.prompts.qa_prompts import get_qa_prompt_for_response_synthesizer
from knowledgeminer.rag.query_engine.create_query_engine import create_query_engine_from_retriever
from knowledgeminer.rag.vector_stores.faiss_vector_store import FaissLamaIndexClient
from llama_index.core.postprocessor import LLMRerank


def create_recursive_retrieval_pipeline(source_data_uri_list: List[str],llm=None, embedding_model=None, embedding_dim=384, sentence_window_chunking=False, load_existing=False) -> BaseRetriever:
    if not load_existing:
        raw_documents = []
        for source_uri in source_data_uri_list:
            # TODO 7: Take the following function as an input param of this method an expect it to return a list of documents always.

            # documents = load_html_content_from_jsonl_field(source_uri)
            documents = load_processed_docupedia_docs_from_jsonl_field(source_uri)
            raw_documents.extend(documents)

        base_nodes = DocumentsToNodesProcessor.docs_to_nodes_sent_chunk_with_title_extraction(raw_documents,
                                                                                              chunk_size=512,
                                                                                              chunk_overlap=32,
                                                                                              llm=llm)
        if not sentence_window_chunking:
            sub_chunks_sizes = [128]
            sub_chunk_overlap = [20]
            all_nodes, all_nodes_id_dict = DocumentsToNodesProcessor.create_index_nodes(base_nodes, sub_chunks_sizes,
                                                                                        sub_chunk_overlap)
        else:
            all_nodes, all_nodes_id_dict = DocumentsToNodesProcessor.create_index_nodes_from_sentence_window_chunking_for_recursive_retrieval(base_nodes,window_size=3)



        faiss_client, vector_index_on_chunks = create_faiss_vector_store(embedding_model, llm,all_nodes,embed_dim=EMBED_DIM)
        faiss_client.save_to_persistent_storage("../out/docupedia_faiss_storage_multi_ling")

        # chromadb_client, vector_index_on_chunks = create_chromadb_vector_store("dummy_vector_store", embedding_model, llm,
        #                                                                     all_nodes)
        # chromadb_client.save_to_persistent_storage("./out/docupedia_chromadb_index")

    else:
        # chromadb_client = ChromaDBLamaIndexClient.load_from_persistent_storage('./out/docupedia_chromadb_index', "dummy_vector_store")
        # vector_index_on_chunks = chromadb_client.get_vector_store_index()

        faiss_client = FaissLamaIndexClient.load_from_persistent_storage("../out/docupedia_faiss_storage_with_bge_base",embedding_model,EMBED_DIM)
        vector_index_on_chunks = faiss_client.get_vector_store_index()

    retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks,"docupedia_retriever", similarity_top_k=5)
    #retriever = vector_index_on_chunks.as_retriever(similarity_top_k=10)
    return retriever


def create_chromadb_vector_store(name, embed_model, llm, nodes):
    chromadb_client = ChromaDBLamaIndexClient(name, embed_model, llm)
    return chromadb_client, chromadb_client.create_vector_store_index(nodes)


def create_faiss_vector_store(embed_model, llm,nodes, embed_dim=1536):
    faiss_vs_client = FaissLamaIndexClient(embed_model,embed_dim, llm)
    return faiss_vs_client, faiss_vs_client.create_vector_store_index(nodes)


def test_run_sentence_window_retrieval(source_data_uri_list, llama_index_llm_client=None):
    for source_uri in source_data_uri_list:
        documents = load_html_content_from_jsonl_field(source_uri)

    llm = llama_index_llm_client.create_basic_azure_openai_client()
    embedding_model = create_basic_azure_openai_embedding_client()

    chunked_nodes = DocumentsToNodesProcessor.create_single_sentence_nodes_with_metadata_window(documents, 3)

    # if you wanted to use OpenAIEmbedding, we should also increase the batch size,
    # since it involves many more calls to the API
    # ctx = ServiceContext.from_defaults(llm=llm, embed_model=OpenAIEmbedding(embed_batch_size=50)), node_parser=node_parser)

    #service_context = create_basic_service_context()
    #sentence_index = VectorStoreIndex(chunked_nodes, service_context=service_context)
    sentence_index = VectorStoreIndex(chunked_nodes)


    query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    #TODO: Add retreiver here to inspect nodes.
    window_response = query_engine.query(
        "What is product market fit?"
    )
    print(window_response)
    window = window_response.source_nodes[0].node.metadata["window"]
    sentence = window_response.source_nodes[0].node.metadata["original_text"]

    print(f"Window: {window}")
    print("------------------")
    print(f"Original Sentence: {sentence}")


def run_know_miner_on_spc():
    Settings.llm = azure_openai_for_llama_index.create_basic_azure_openai_client()
    Settings.embed_model = create_basic_azure_openai_embedding_client()
    EMBED_DIM = 1536

    with open(
            "/Users/gar1syv/Documents/git_repos/bt-system-planner-copilot-data-science/src/predict/filtered_ordering_information.json",
            'r') as fp:
        prod_cat = fp.read()
    prod_cat_list = json.loads(prod_cat)

    nodes = []
    for product in prod_cat_list:
        nodes.append(Document(text=product["product group"] + " : " +product["Ordering Info (Long Description) de"],
                              metadata={
                                  "VEPOS": product["VEPOS"],
                                  "Assetkey": product["Assetkey"],
                                  "product group": product["product group"]
                              }))

    faiss_vs_client = FaissLamaIndexClient(Settings.embed_model, EMBED_DIM, Settings.llm)

    faiss_index = faiss_vs_client.create_vector_store_index(nodes)
    faiss_vs_client.save_to_persistent_storage("./out/bt_product_catalog_faiss_storage")

    vector_retriever = faiss_index.as_retriever(similarity_top_k=7)
#     results = vector_retriever.retrieve("""Erweiterung der vorhandenen Bosch-
# Brandmeldezentrale Typ FPA
# für 256 Adressen, 5 Ringmodule, sowie
# Gehäuseerweiterung für 10 Funktionsmodule.
# Die Verkabelung der Brandmeldeanlage wird von der
# Ausführungsfirma Elt vorgenommen.
# Samtliche notwendige Abstimmungen sind mit den
# Einheitspreisen abgegolten.
# Vorhandene, modulare Brandmeldezentrale vom Nutzer
# übernehmen,reinigen, prüfen,
# am vorgesehenen Installationsort montieren und anschließen,
# einschließlich Softwareupdate.""")
#
#     for node in results:
#         print(str(node.metadata["Assetkey"]) + " : " + str(node.metadata["VEPOS"]))
#
#     print(results)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=3, verbose=True
    )

    retriever = get_fusion_retriever_with_bm25(vector_retriever, bm25_retriever, top_k=10)

    while True:
        query = input("Enter query: ")

        results = retriever.retrieve(query)
        for node in results:
            print(str(node.metadata["Assetkey"]) + " : " + str(node.metadata["VEPOS"]))
        print(results)


if __name__ == "__main__":
    run_know_miner_on_spc()

    # # docupedia_docs = load_html_content_from_jsonl_field("/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl")
    # # print(len(docupedia_docs))
    # # print(docupedia_docs[0])
    #
    # #knowledge_source_uri_list = ['/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl']
    # knowledge_source_uri_list = ["/Users/gar1syv/Documents/git_repos/KnowledgeMiner/out/parsed_docupedia_sources.jsonl"]
    # #TODO 8: Try to extract EMBED_DIM from model config/properties
    # Settings.llm = azure_openai_for_llama_index.create_basic_azure_openai_client()
    # Settings.embed_model = create_basic_azure_openai_embedding_client()
    # EMBED_DIM = 1536
    #
    # #TODO: Use bge-large or bge-base embedding model: "BAAI/bge-base-en-v1.5"
    # # TODO: Provide multi-lingual embedding model and check llama3.1 for the same - else ask llama3 to translate full context to english first before answering.
    # # Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    # #                                                tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # # Settings.embed_model = create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5")
    # # EMBED_DIM = 384
    #
    # retriever = create_recursive_retrieval_pipeline(knowledge_source_uri_list,Settings.llm,Settings.embed_model,sentence_window_chunking=True,load_existing=False)
    #
    # response_synthesizer = get_retrieved_context_response_synthesizer(mode=ResponseMode.COMPACT, structured_answer_filtering=False, qa_prompt=get_qa_prompt_for_response_synthesizer())
    #
    # query_engine = create_query_engine_from_retriever(retriever, response_synthesizer)
    # # Done: 2.1: Implement and compare other alternatives to response_synthesizer: e.g. query_engine or direct LLM call
    # # TODO 2.2: Implement prompt-engineering for all the above methods
    # # TODO 3: Implement storing and loading of persistent index
    # # Done 4: Implement local embedding
    # # TODO 5: Translate non-english into english before chunking
    # # TODO 6: Implement evaluation for above methods.
    #
    # while True:
    #     query = input("Enter your query:")
    #
    #     # context_nodes = retriever.retrieve(query)
    #     # response = response_synthesizer.synthesize(query,nodes=context_nodes)
    #
    #     response = query_engine.query(query)
    #     print(response.source_nodes)
    #     print(response)


