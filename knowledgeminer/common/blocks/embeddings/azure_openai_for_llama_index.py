import os
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding


def create_basic_azure_openai_embedding_client(model_name: str = "text-embedding-ada-002",
                                               api_version: str = "2023-03-15-preview") -> AzureOpenAIEmbedding:
    # You need to deploy your own embedding model as well as your own chat completion model
    embedding_model = AzureOpenAIEmbedding(
        model=model_name,
        # deployment_name="text-davinci-003",
        api_key=os.environ.get("OPENAI_API_KEY", "Please set this value"),
        azure_endpoint=os.environ.get("OPENAI_API_BASE", "Please set this value"),
        api_version=api_version,
    )
    return embedding_model
