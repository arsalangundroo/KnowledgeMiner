import os
#TODO: change from chat model to normal model
from langchain_community.chat_models import AzureChatOpenAI


def create_basic_azure_openai_client(openai_api_version="2023-03-15-preview", deployment_name="gpt35"):
    return AzureChatOpenAI(
        azure_endpoint=os.environ.get("OPENAI_API_KEY", "Please set this value"),
        openai_api_key=os.environ.get("OPENAI_API_BASE", "Please set this value"),
        openai_api_version=openai_api_version,
        deployment_name=deployment_name,
        temperature=0.0,
    )
