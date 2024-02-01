from llama_index.response_synthesizers import get_response_synthesizer, BaseSynthesizer
from knowledgeminer.common.utils.service_context_handler import create_basic_service_context


def get_retrieved_context_response_synthesizer(service_context=None,mode="refine"):
    #TODO: read mode from config.
    if service_context is None:
        service_context = create_basic_service_context()
    response_synthesizer = get_response_synthesizer(service_context=service_context,response_mode=mode)
    return response_synthesizer
