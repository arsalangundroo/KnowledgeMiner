import os

from langchain_openai import AzureOpenAIEmbeddings
def create_basic_azure_openai_embedding_client(model_name: str = "text-embedding-ada-002",
                                               api_version: str = "2023-03-15-preview") -> AzureOpenAIEmbeddings:
    # You need to deploy your own embedding model as well as your own chat completion model
    embedding_model = AzureOpenAIEmbeddings(
        model=model_name,
        # deployment_name="text-davinci-003",
        api_key=os.environ.get("OPENAI_API_KEY", "Please set this value"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "Please set this value"),
        api_version=api_version,
    )
    return embedding_model
