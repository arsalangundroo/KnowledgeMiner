from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer
from llama_index.core.response_synthesizers import ResponseMode


def get_retrieved_context_response_synthesizer(mode=ResponseMode.REFINE, structured_answer_filtering=False,qa_prompt=None,llm=None):
    if llm:
        answerer = llm
    else:
        answerer = Settings.llm

    response_synthesizer = get_response_synthesizer(response_mode=mode,structured_answer_filtering=structured_answer_filtering,simple_template=qa_prompt, llm=answerer)
    return response_synthesizer
