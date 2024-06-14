from configparser import ConfigParser

from knowledgeminer.agents.agents import create_zero_shot_react_langchain_agent
from knowledgeminer.agents.tools.vector_store_retrieval_tools import LlamaIndexSourceBookRetrievalTool
from knowledgeminer.rag.loaders.pdf_loaders import PDFReader
from knowledgeminer.common.blocks.llm import \
    azure_openai_for_llama_index as llama_index_llm_client  # import create_basic_azure_openai_client as
from knowledgeminer.common.blocks.llm import \
    azure_openai_for_langchain as langchain_llm_client
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.rag.preprocessors.node_processors import DocumentsToNodesProcessor
from knowledgeminer.rag.retrievers.recursive_retrieval import createRecursiveRetrieverFromIndex
from knowledgeminer.rag.vector_stores.chroma_db import ChromaDBLamaIndexClient


def get_data_sources_from_config(config):
    # print(config["DocumentSources"])
    data_sources = config["DocumentSources"]
    if data_sources is None:
        return None
    doc_source_uri_list = list(data_sources.values())
    return doc_source_uri_list


def get_chunking_strategy_from_config(config):
    chunking_strategy = config["Chunking"]
    if chunking_strategy["chunking_strategy"] == "docs_to_nodes_sent_chunk_with_title_extraction":
        chunk_func = DocumentsToNodesProcessor.docs_to_nodes_sent_chunk_with_title_extraction
    chunk_sizes = [int(x.strip()) for x in chunking_strategy["chunk_sizes"].split(",")]
    chunk_overlaps = [int(x.strip()) for x in chunking_strategy["chunk_overlaps"].split(",")]

    assert len(chunk_sizes) == len(chunk_overlaps)
    print(chunk_sizes)

    return chunk_func, chunk_sizes, chunk_overlaps


def get_vector_store_from_config(config):
    vs_type = config["VectorStore"]["vector_store_type"]
    vs_name = config["VectorStore"]["vector_store_name"]
    embedding_model = config["VectorStore"]["embedding_model"]
    llm = config["VectorStore"]["llm"]

    if embedding_model == "basic_azure_openai":
        embedding_model = create_basic_azure_openai_embedding_client()
    else:
        NotImplementedError("Given embedding model not implemented!")
    if llm == "basic_azure_openai":
        llm = llama_index_llm_client.create_basic_azure_openai_client()
    else:
        NotImplementedError("Given LLM client not implemented!")

    if vs_type == "chromaDB":
        chromadb_client = ChromaDBLamaIndexClient(vs_name, embedding_model, llm)
    else:
        NotImplementedError("Given VectorStore client not implemented!")
        return None

    return chromadb_client

def get_retriever_initialization_params_from_config(config):
    if config["Retriever"]["retriever_type"] == "recursive_retriever":
        return createRecursiveRetrieverFromIndex, config["Retriever"]["retriever_name"], config["Retriever"]["top_k"]


def build_indexing_pipeline(config):
    doc_sources = get_data_sources_from_config(config)
    knowledge_source_uri_list = get_data_sources_from_config(config)
    print(knowledge_source_uri_list)

    # Load Docs
    # TODO: Implement proper loading through a common interface for all loaders
    raw_documents = []
    for doc_source_uri in knowledge_source_uri_list:
        documents = PDFReader.load_pdf(filename=doc_source_uri)
        raw_documents.extend(documents)

    default_llm = llama_index_llm_client.create_basic_azure_openai_client()
    # embedding_model = create_basic_azure_openai_embedding_client()

    # Chunking
    # TODO: Implement proper chunking through a common interface for all chunking methods
    chunking_function, chunk_sizes, chunk_overlaps = get_chunking_strategy_from_config(config)
    base_nodes = chunking_function(raw_documents[:5], llm=default_llm)
    print(base_nodes)
    #TODO: FIX following line for generic pipeline run
    all_nodes, all_nodes_id_dict = DocumentsToNodesProcessor.create_index_nodes(base_nodes, chunk_sizes,
                                                                                chunk_overlaps)

    # Indexing
    vector_store_client = get_vector_store_from_config(config)
    vector_index_on_chunks = vector_store_client.create_vector_store_index(all_nodes)

    # vector_index_on_chunks = create_vector_store_index(all_nodes, llm, embedding_model)
    # chromadb_client, vector_index_on_chunks = create_chromadb_vector_store("dummy_vector_store", embedding_model, llm,all_nodes)
    # faiss_client, vector_index_on_chunks = create_faiss_vector_store(embedding_model, llm,all_nodes)
    # faiss_client.save_to_persistent_storage("./faiss_storage")
    # print(vector_index_on_chunks.service_context)

    return vector_index_on_chunks, all_nodes_id_dict


def create_retriever(vector_index_on_chunks, all_nodes_id_dict):
    create_retriever_func, retriever_name, top_k = get_retriever_initialization_params_from_config(config)
    retriever = create_retriever_func(vector_index_on_chunks, all_nodes_id_dict, retriever_name, similarity_top_k=top_k)
    return retriever


def create_retrieval_tool_for_agent(name, description, retriever, response_synthesizer):
    tool = LlamaIndexSourceBookRetrievalTool(name=name, description=description, retriever=retriever,
                                             response_synthesizer=response_synthesizer)
    return tool


def create_agent(tools):
    llm = langchain_llm_client.create_basic_azure_openai_client()
    return create_zero_shot_react_langchain_agent(llm, tools=tools)


if __name__ == "__main__":
    # del os.environ['OPENAI_API_BASE']
    config = ConfigParser()
    config.read('/Users/gar1syv/Documents/git_repos/KnowledgeMiner/knowledgeminer/configs/indexing_pipeline_config.ini')
    vector_index_on_chunks, all_nodes_id_dict = build_indexing_pipeline(config)
    retriever = create_retriever(vector_index_on_chunks, all_nodes_id_dict)
    query = input("Enter a query:")
    retrieved_context = retriever.retrieve(query)
    print([rc.text for rc in retrieved_context])
    print(len(retrieved_context))


    # TODO: Make below portion configurable later.
    # product_management_tool = create_retrieval_tool_for_agent(name="product_management_tool",
    #                                                     description="searches the book of product management regarding a specific topic or "
    #                                                                 "query.",
    #                                                     retriever=retriever,
    #                                                   #  response_synthesizer=response_synthesizer)
    #                                                     )
    # tools = [product_management_tool]
    # agent = create_agent(tools)
    #
    # query = input("Enter your query:")
    # agent.run(query)

