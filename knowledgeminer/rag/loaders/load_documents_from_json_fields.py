import json
from typing import List

from tqdm import tqdm
from llama_index.core import Document

from knowledgeminer.common.utils.translate_to_english import OpenAITranslator
from knowledgeminer.rag.preprocessors.html_parsing import parse_html_text_into_nodes
#from googletrans import Translator
import re

#lang_detector = Translator()
translator = OpenAITranslator()



def remove_extra_spaces(text):
    return re.sub('\s+', ' ', text).strip()

def load_html_content_from_jsonl_field(jsonl_file_name: str) -> List[Document]:
    docupedia_page_docs = []
    parsed_json_list = []
    with open(jsonl_file_name, "r") as fp:
        json_list = list(fp)

    for idx,json_str in tqdm(enumerate(json_list)):
        try:
            json_obj = json.loads(json_str)
            html_doc = Document(text=json_obj["html_content"],
                                metadata={
                                    "docupedia_page_id": json_obj["page_id"],
                                    "docupedia_url": json_obj["url"]
                                })

            #html_doc_nodes.extend(parse_html_text_into_nodes(html_doc))
            html_doc_nodes = parse_html_text_into_nodes(html_doc)
            parsed_text = ""
            for node in html_doc_nodes:
                clean_text = remove_extra_spaces(node.text)
                if clean_text and len(clean_text)>0:
                    #TODO 8: Implement this lang detection over entire page and for dual language texts
                    max_len = min(len(clean_text),500)
                    detected = lang_detector.detect(clean_text[:max_len])
                    if detected.lang == "en":
                        parsed_text += node.text
                    elif detected.lang == "de":
                        parsed_text += translator.tranlsate_german_to_english_with_gpt(clean_text)
                    else:
                        print(f"New language [{detected.lang}] detected for page_id: {json_obj['page_id']}")

            docupedia_page_docs.append(
                Document(text=parsed_text,
                         metadata={
                             "docupedia_page_id": json_obj["page_id"],
                             "docupedia_url": json_obj["url"]
                         }))

            parsed_json_list.append({
                "text": parsed_text,
                "docupedia_page_id": json_obj["page_id"],
                "docupedia_url": json_obj["url"]
            })
            if len(parsed_json_list)%10==0:
                with open("../out/parsed_docupedia_sources.jsonl", 'w') as fp:
                    for item in parsed_json_list:
                        fp.write(json.dumps(item) + "\n")

        except Exception as e:
            print(e)
            print(f"Error with page_id: {json_obj['page_id']}")
            with open("../out/error_page_ids.txt", 'a') as fp:
                    fp.write(json_obj['page_id']+"\n")

    with open("../out/parsed_docupedia_sources.jsonl", 'w') as fp:
        for item in parsed_json_list:
            fp.write(json.dumps(item) + "\n")
    return docupedia_page_docs


# def translate_to_english(original_text, original_lang) -> str:
#
#     if original_lang == "de":
#         translated_text = ""
#         original_sentences = original_text.split(".")
#         for sent in original_sentences:
#             if len(sent.strip())>0:
#                 print(f"=================\n{sent}\n+++++++++++++++++++++")
#                 translated = lang_detector.translate(sent, dest="en")
#                 assert len(translated.text) > 0
#                 translated_text+=translated_text+". "
#
#         return translated_text


def load_processed_docupedia_docs_from_jsonl_field(jsonl_file_name: str) -> List[Document]:
    docupedia_page_docs = []

    with open(jsonl_file_name, "r") as fp:
        json_list = list(fp)

    for idx, json_str in tqdm(enumerate(json_list)):
        try:
            json_obj = json.loads(json_str)
            docupedia_doc = Document(text=json_obj["text"],
                                metadata={
                                    "docupedia_page_id": json_obj["docupedia_page_id"],
                                    "docupedia_url": json_obj["docupedia_url"]
                                })
            docupedia_page_docs.append(docupedia_doc)

        except Exception as e:
            print(e)
            print(f"Error with page_id: {json_obj['docupedia_page_id']}")

    return docupedia_page_docs


if __name__=='__main__':

    load_html_content_from_jsonl_field('/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl')
