from llama_index.core import PromptTemplate


def get_qa_prompt_for_response_synthesizer():
    qa_prompt_tmpl = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Please also keep the answer concise in one or two sentences strictly.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)
    return qa_prompt

