import json
from tqdm import tqdm
from llama_index.core import Document
from knowledgeminer.rag.preprocessors.html_parsing import parse_html_text_into_nodes


def load_html_content_from_jsonl_field(jsonl_file_name:str):
    html_doc_nodes = []
    with open(jsonl_file_name,"r") as fp:
        json_list = list(fp)

    for json_str in tqdm(json_list):
        json_obj=json.loads(json_str)
        html_doc = Document(text=json_obj["html_content"])
        #print(type(hdoc))
        # print(hdoc)
        # print(hdoc.id_)

        # html_doc_nodes.append(parse_html_text_into_nodes(json_obj["html_content"],parent_metadata ={
        #     "docupedia_page_id": json_obj["page_id"],
        #     "docupedia_url": json_obj["url"]}))
        html_doc_nodes.append(parse_html_text_into_nodes(html_doc))

    return html_doc_nodes
