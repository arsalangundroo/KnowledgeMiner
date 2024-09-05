from knowledgeminer.rag.loaders.load_documents_from_json_fields import load_html_content_from_jsonl_field


if __name__ == "__main__":

    html_doc_nodes = load_html_content_from_jsonl_field("/Users/gar1syv/Documents/ask_bosch_data/ngw.jsonl")
    print(len(html_doc_nodes))
    print(html_doc_nodes[0])