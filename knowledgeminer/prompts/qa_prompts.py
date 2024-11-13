from llama_index.core import PromptTemplate

BASE_PROMPT = \
    "Context information is below.\n" \
    "---------------------\n" \
    "{context_str}\n" \
    "---------------------\n"  \
    "Given the context information and not prior knowledge, " \
    "answer the query.\n" \
    "Please also keep the answer concise in one or two sentences strictly.\n" \
    "Query: {query_str}\n" \
    "Answer: "


MULTI_LANG_PROMPT =  \
    "Context information is below. Be mindful that the provided context can be either in English or in German language.\n" \
    "---------------------\n" \
    "{context_str}\n" \
    "---------------------\n" \
    "Given the context information and not prior knowledge, " \
    "answer the query.\n" \
    "Please also keep the answer concise in one or two sentences strictly. Make sure to answer only in English language\n" \
    "Query: {query_str}\n" \
    "Answer: "

DETAILED_MULTI_LANG_PROMPT_v1 =  \
    "You are a an assistant to employees who provides information from the documentation of an Enterprise platform and its various components." \
    "The employees will ask you query about the platform or its components and operations.\n" \
    "The documentation relevant to their question will be provided to you as context information" \
    "Context information is below. Be mindful that the provided context can be either in English or in German language.\n" \
    "---------------------\n" \
    "{context_str}\n" \
    "---------------------\n" \
    "Based on this the context information and not prior knowledge, " \
    "answer the user's query.\n" \
    "\nQuery: {query_str}\n" \
    "Please also keep the answer concise in one or two sentences strictly. Make sure to answer only in English language.\n" \
    "Answer: "


DETAILED_MULTI_LANG_PROMPT_v2 =  \
    "You are a an assistant to employees who provides information from the documentation of an Enterprise platform and its various components. \n" \
    "The employees will ask you a query.\n" \
    "The documents relevant to their query will be provided to you as context information.\n" \
    "Please keep the answer concise in one or two sentences strictly. Make sure to answer only in English language.\n" \
    "Context information is given below, enclosed in \'-\' characters. Be mindful that the provided context can be either in English or in German language and make sure you understand both.\n" \
    "---------------------\n" \
    "{context_str}\n" \
    "---------------------\n" \
    "Based on this the context information and not prior knowledge, " \
    "answer the user's query.\n" \
    "\nQuery: {query_str}\n" \
    "Answer: "


DETAILED_MULTI_LANG_PROMPT_v3 =  \
    "You are a an assistant to employees who provides information from the documentation of an Enterprise platform and its various components. \n" \
    "The employees will ask you a query.\n" \
    "The documents relevant to their query will be provided to you as context information.\n" \
    "Please keep the answer concise in one or two sentences strictly. Make sure to answer only in English language.\n" \
    "Context information is given below, enclosed in \'-\' characters. Be mindful that the provided context can be either in English or in German language and make sure you understand both.\n" \
    "---------------------\n" \
    "{context_str}\n" \
    "---------------------\n" \
    "Based on this the context information and not prior knowledge, " \
    "answer the user's query. All the provided context information may not be relevant to the query. \n" \
    "Only use the relevant parts of the context information to formulate a precise and concise answer.\n" \
    "\nQuery: {query_str}\n" \
    "Answer: "


def get_qa_prompt_for_response_synthesizer(prompt_template_str=BASE_PROMPT):
    qa_prompt_tmpl = (prompt_template_str)
    qa_prompt = PromptTemplate(qa_prompt_tmpl)
    return qa_prompt

