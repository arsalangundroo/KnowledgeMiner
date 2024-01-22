from llama_index.response_synthesizers import get_response_synthesizer, BaseSynthesizer


def get_retrieved_context_response_synthesizer(mode="refine"):
    #TODO: read mode from config.
    response_synthesizer = get_response_synthesizer(response_mode=mode)
    return response_synthesizer
