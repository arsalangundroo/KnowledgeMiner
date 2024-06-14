import os
from typing import List
from llmsherpa.readers import LayoutPDFReader

from knowledgeminer.agents.agents import create_zero_shot_react_langchain_agent
from knowledgeminer.agents.tools.vector_store_retrieval_tools import LlamaIndexSourceBookRetrievalTool
from knowledgeminer.common.utils.service_context_handler import create_basic_service_context
from knowledgeminer.rag.loaders.pdf_loaders import PDFReader
from knowledgeminer.rag.postprocessors.retrieval_response_synthesizer import get_retrieved_context_response_synthesizer
from knowledgeminer.rag.preprocessors.node_processors import DocumentsToNodesProcessor
from knowledgeminer.common.blocks.llm import \
    azure_openai_for_llama_index as llama_index_llm_client  # import create_basic_azure_openai_client as
from knowledgeminer.common.blocks.llm import \
    azure_openai_for_langchain as langchain_llm_client  # import create_basic_azure_openai_client
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.rag.vector_stores.default_LI_vector_store import create_vector_store_index
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from llama_index.retrievers import BaseRetriever
from knowledgeminer.rag.vector_stores.chroma_db import ChromaDBLamaIndexClient
from knowledgeminer.rag.vector_stores.faiss_vector_store import FaissLamaIndexClient
from llama_index import set_global_service_context, VectorStoreIndex
from llama_index.postprocessor import MetadataReplacementPostProcessor


def create_retrieval_pipeline(source_data_uri_list: List[str]) -> BaseRetriever:
    sub_chunks_sizes = [128]
    sub_chunk_overlap = [20]
    raw_documents = []
    for source_uri in source_data_uri_list:
        documents = PDFReader.load_pdf(filename=source_uri)
        raw_documents.extend(documents)

    llm = llama_index_llm_client.create_basic_azure_openai_client()
    embedding_model = create_basic_azure_openai_embedding_client()

    base_nodes = DocumentsToNodesProcessor.docs_to_nodes_sent_chunk_with_title_extraction(raw_documents[:5], llm=llm)
    all_nodes, all_nodes_id_dict = DocumentsToNodesProcessor.create_index_nodes(base_nodes, sub_chunks_sizes,
                                                                                sub_chunk_overlap)
    #vector_index_on_chunks = create_vector_store_index(all_nodes, llm, embedding_model)
    chromadb_client, vector_index_on_chunks = create_chromadb_vector_store("dummy_vector_store_6", embedding_model, llm, all_nodes)
    # faiss_client, vector_index_on_chunks = create_faiss_vector_store(embedding_model, llm,all_nodes)
    # faiss_client.save_to_persistent_storage("./faiss_storage")
    #print(vector_index_on_chunks.service_context)
    retriever = createRecursiveRetrieverFromIndex(vector_index_on_chunks, all_nodes_id_dict,"dummy_retriever", similarity_top_k=3)
    return retriever


def create_chromadb_vector_store(name, embed_model, llm, nodes):
    chromadb_client = ChromaDBLamaIndexClient(name, embed_model, llm)
    return chromadb_client, chromadb_client.create_vector_store_index(nodes)

def create_faiss_vector_store(embed_model, llm,nodes):
    faiss_vs_client = FaissLamaIndexClient(embed_model, llm)
    return faiss_vs_client, faiss_vs_client.create_vector_store_index(nodes)
def create_retrieval_tool_for_agent(name, description, retriever, response_synthesizer):
    tool = LlamaIndexSourceBookRetrievalTool(name=name, description=description, retriever=retriever,
                                             response_synthesizer=response_synthesizer)
    return tool


def create_agent(tools):
    llm = langchain_llm_client.create_basic_azure_openai_client()
    return create_zero_shot_react_langchain_agent(llm, tools=tools)


def test_run_sentence_window_retrieval(source_data_uri_list):
    for source_uri in source_data_uri_list:
        documents = PDFReader.load_pdf(filename=source_uri)

    llm = llama_index_llm_client.create_basic_azure_openai_client()
    embedding_model = create_basic_azure_openai_embedding_client()

    chunked_nodes = DocumentsToNodesProcessor.create_single_sentence_nodes_with_metadata_window(documents[:100], 3)

    # if you wanted to use OpenAIEmbedding, we should also increase the batch size,
    # since it involves many more calls to the API
    # ctx = ServiceContext.from_defaults(llm=llm, embed_model=OpenAIEmbedding(embed_batch_size=50)), node_parser=node_parser)

    service_context = create_basic_service_context()
    sentence_index = VectorStoreIndex(chunked_nodes, service_context=service_context)

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


if __name__ == "__main__":
    #del os.environ['OPENAI_API_BASE']

    knowledge_source_uri_list = ['/Users/gar1syv/Desktop/books/The_Product_Book_2nd_Edition.pdf']
    retriever = create_retrieval_pipeline(knowledge_source_uri_list)
    response_synthesizer = get_retrieved_context_response_synthesizer(mode="refine")

    # # quran_search_tool = create_retrieval_tool_for_agent(name="quran_query_tool",
    # #                                                     description="searches the book of Quran for verses ("
    # #                                                                 "information) regarding a specific topic or "
    # #                                                                 "query.",
    # #                                                     retriever=quran_retriever,
    # #                                                     response_synthesizer=response_synthesizer)
    # # tools = [quran_search_tool]
    #
    product_management_tool = create_retrieval_tool_for_agent(name="product_management_tool",
                                                        description="searches the book of product management regarding a specific topic or "
                                                                    "query.",
                                                        retriever=retriever,
                                                        response_synthesizer=response_synthesizer)
    tools = [product_management_tool]
    agent = create_agent(tools)

    query = input("Enter your query:")
    agent.run(query)

    # llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    # pdf_url = "/Users/gar1syv/Desktop/books/Quran/quran-in-modern-english.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
    # pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    # print("reading pdf")
    # doc = pdf_reader.read_pdf(pdf_url)
    # print(len(doc.json))
    # for json_element in doc.json:
    #     if 'sentences' in json_element and len(json_element['sentences'])>1:
    #         print(len(json_element['sentences']))
    # print(doc.json)

    #faiss_index_client = FaissLamaIndexClient.load_from_persistent_storage("../../faiss_storage")
    #print(type(faiss_index_client))

    # chroma_loaded_index_client = ChromaDBLamaIndexClient.load_from_persistent_storage("./chroma_db_storage/", "dummy_vector_store")
    #
    # print(type(chroma_loaded_index_client))
    # set_global_service_context(create_basic_service_context())
    # # chroma_loaded_index_client._index
    # query_engine = chroma_loaded_index_client.get_vector_store_index().as_query_engine(service_context=create_basic_service_context())
    # response = query_engine.query("What is product market fit?")
    # print(response)

    #test_run_sentence_window_retrieval(knowledge_source_uri_list)
