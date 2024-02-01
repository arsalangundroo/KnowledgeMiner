from llama_index import ServiceContext
from knowledgeminer.common.blocks.llm import azure_openai_for_llama_index
from knowledgeminer.common.blocks.embeddings.azure_openai_for_llama_index import create_basic_azure_openai_embedding_client


def create_basic_service_context():
    llm = azure_openai_for_llama_index.create_basic_azure_openai_client()
    embedding_model = create_basic_azure_openai_embedding_client()
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding_model,
    )

    return service_context
