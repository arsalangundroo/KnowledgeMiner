import os
from llama_index.llms import AzureOpenAI


def create_basic_azure_openai_client(model_name: str = "gpt-35-turbo-16k", deployment_name: str = "gpt35",
                                     api_version: str = "2023-03-15-preview") -> AzureOpenAI:
    llm = AzureOpenAI(
        model=model_name,
        deployment_name=deployment_name,
        api_key=os.environ.get("OPENAI_API_KEY", "Please set this value"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "Please set this value"),
        api_version=api_version,
    )
    return llm




