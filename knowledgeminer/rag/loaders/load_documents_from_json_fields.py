import json
from typing import List

from tqdm import tqdm
from llama_index.core import Document
from knowledgeminer.rag.preprocessors.html_parsing import parse_html_text_into_nodes


def load_html_content_from_jsonl_field(jsonl_file_name:str)->List[Document]:
    docupedia_page_docs = []
    with open(jsonl_file_name,"r") as fp:
        json_list = list(fp)

    for json_str in tqdm(json_list):
        json_obj=json.loads(json_str)
        html_doc = Document(text=json_obj["html_content"],
                            metadata={
                                    "docupedia_page_id": json_obj["page_id"],
                                    "docupedia_url": json_obj["url"]
                            })

        #html_doc_nodes.extend(parse_html_text_into_nodes(html_doc))
        html_doc_nodes = parse_html_text_into_nodes(html_doc)
        parsed_text=""
        for node in html_doc_nodes:
            parsed_text += node.text

        docupedia_page_docs.append(
            Document(text=parsed_text,
                     metadata={
                         "docupedia_page_id": json_obj["page_id"],
                         "docupedia_url": json_obj["url"]
                     }))

    return docupedia_page_docs
