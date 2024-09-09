import os
from typing import List
import json

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.rag.loaders.load_documents_from_json_fields import load_html_content_from_jsonl_field
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.preprocessors.node_processors import DocumentsToNodesProcessor
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.chroma_db import ChromaDBLamaIndexClient
from llama_index.core.response_synthesizers import ResponseMode
from knowledgeminer.common.blocks.llm import azure_openai_for_llama_index, meta_llama3
from llama_index.llms.azure_openai import AzureOpenAI
from knowledgeminer.common.blocks.embeddings.hf_embed_models import create_hf_embed_model
from knowledgeminer.prompts.qa_prompts import get_qa_prompt_for_response_synthesizer
from knowledgeminer.rag.query_engine.create_query_engine import create_query_engine_from_retriever


def create_retrieval_pipeline(source_data_uri_list: List[str],llm=None, embedding_model=None) -> BaseRetriever:
    sub_chunks_sizes = [128]
    sub_chunk_overlap = [20]
    raw_documents = []
    for source_uri in source_data_uri_list:
        documents = load_html_content_from_jsonl_field(source_uri)
        raw_documents.extend(documents)


    base_nodes = DocumentsToNodesProcessor.docs_to_nodes_sent_chunk_with_title_extraction(raw_documents, llm=llm)
    all_nodes, all_nodes_id_dict = DocumentsToNodesProcessor.create_index_nodes(base_nodes, sub_chunks_sizes,
                                                                                sub_chunk_overlap)
    # faiss_client, vector_index_on_chunks = create_faiss_vector_store(embedding_model, llm,all_nodes)
    # faiss_client.save_to_persistent_storage("./faiss_storage")
    chromadb_client, vector_index_on_chunks = create_chromadb_vector_store("dummy_vector_store", embedding_model, llm,
                                                                           all_nodes)
    chromadb_client.save_to_persistent_storage("./out/docupedia_chromadb_index")
    with open("./out/docupedia_all_nodes_id_dict.json",'w') as fp:
        json.dumps(all_nodes_id_dict,fp)
    #print(vector_index_on_chunks.service_context)
    retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks, all_nodes_id_dict,"docupedia_retriever", similarity_top_k=3)
    return retriever


def create_chromadb_vector_store(name, embed_model, llm, nodes):
    chromadb_client = ChromaDBLamaIndexClient(name, embed_model, llm)
    return chromadb_client, chromadb_client.create_vector_store_index(nodes)


def create_faiss_vector_store(embed_model, llm,nodes):
    faiss_vs_client = FaissLamaIndexClient(embed_model, llm)
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


def create_recursive_retriever_from_persistent_index(persist_url,collection_name):
    chromadb_client = ChromaDBLamaIndexClient.load_from_persistent_storage(persist_url,collection_name)
    with open("./out/docupedia_all_nodes_id_dict.json",'r') as fp:
        all_nodes_id_dict = json.loads(fp)
    retriever = createRecursiveRetrieverFromIndex(chromadb_client.get_vector_store_index(), all_nodes_id_dict, "docupedia_retriever",
                                                  similarity_top_k=3)
    return retriever


if __name__ == "__main__":

    # docupedia_docs = load_html_content_from_jsonl_field("/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl")
    # print(len(docupedia_docs))
    # print(docupedia_docs[0])

    knowledge_source_uri_list = ['./data/ngw.jsonl']

    # Settings.llm = azure_openai_for_llama_index.create_basic_azure_openai_client()
    # Settings.embed_model = create_basic_azure_openai_embedding_client()

    # TODO 1: Provide a configured LLM to the response synthesizer
    Settings.llm = meta_llama3.create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                                                   tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct")
    Settings.embed_model = create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5")

    retriever = create_retrieval_pipeline(knowledge_source_uri_list,Settings.llm,Settings.embed_model)
    #retriever = create_recursive_retriever_from_persistent_index("./out/docupedia_chromadb_index","dummy_vector_store")

    response_synthesizer = get_retrieved_context_response_synthesizer(mode=ResponseMode.COMPACT, structured_answer_filtering=False, qa_prompt=get_qa_prompt_for_response_synthesizer())

    query_engine = create_query_engine_from_retriever(retriever, response_synthesizer)
    # Done: 2.1: Implement and compare other alternatives to response_synthesizer: e.g. query_engine or direct LLM call
    # TODO 2.2: Implement prompt-engineering for all the above methods
    # TODO 3: Implement storing and loading of persistent index
    # Done 4: Implement local embedding
    # TODO 5: Translate non-english into english before chunking
    # TODO 6: Implement evaluation for above methods.

    while True:
        query = input("Enter your query:")

        # context_nodes = retriever.retrieve(query)
        # response = response_synthesizer.synthesize(query,nodes=context_nodes)

        response = query_engine.query(query)

        print(response)


