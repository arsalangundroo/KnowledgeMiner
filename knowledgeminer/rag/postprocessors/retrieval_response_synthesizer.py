from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer
#from knowledgeminer.common.utils.service_context_handler import create_basic_service_context
from llama_index.core.response_synthesizers import ResponseMode


def get_retrieved_context_response_synthesizer(mode=ResponseMode.REFINE, structured_answer_filtering=False,qa_prompt=None):
    #TODO: read mode from config.
    # if service_context is None:
    #     service_context = create_basic_service_context()

    response_synthesizer = get_response_synthesizer(response_mode=mode,structured_answer_filtering=structured_answer_filtering,qa_prompt=qa_prompt)
    return response_synthesizer
