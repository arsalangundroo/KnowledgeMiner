import os
from llama_index.llms.azure_openai import AzureOpenAI
#
#
# def create_basic_azure_openai_client(model_name: str = "gpt-35-turbo-16k", deployment_name: str = "gpt35",
#                                      api_version: str = "2023-03-15-preview") -> AzureOpenAI:
#     llm = AzureOpenAI(
#         model=model_name,
#         deployment_name=deployment_name,
#         api_key=os.environ.get("OPENAI_API_KEY", "Please set this value"),
#         azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "Please set this value"),
#         api_version=api_version,
#     )
#     return llm

def create_basic_azure_openai_client(model="gpt-4-1106-preview",openai_api_version="2024-04-01-preview", deployment_name="gpt-4-1106-preview"):

    return AzureOpenAI(
        model=model,
        deployment_name=deployment_name,
        api_key= os.environ.get("OPENAI_API_KEY", "Please set this value"),
        azure_endpoint= os.environ.get("AZURE_OPENAI_ENDPOINT", "Please set this value"),
        api_version=openai_api_version,
    )





