from abc import ABC
from typing import Optional
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from llama_index.core.base_retriever import BaseRetriever as BaseRetrieverLI
from llama_index.response_synthesizers import BaseSynthesizer
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class LlamaIndexSourceBookRetrievalTool(BaseTool):
    #TODO: check for ABC class definition and structure of attributes here
    # TODO: define inputs of the tool as tool_input class.
    name: str
    description: str
    retriever: BaseRetrieverLI
    response_synthesizer: BaseSynthesizer

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None,
            # metadata_args: dict[str,str]=None
    ) -> str:
        """Use the tool."""
        # TODO: (1) define response schema to preprocess/filter output as consumable for Agent
        # (2) Handle multiple answers/contexts.
        # (3) check async get_ method on retriever
        context_nodes = self.retriever.retrieve(query)
        return self.response_synthesizer.synthesize(query, context_nodes)
        # return self.retriever.retrieve(query)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")