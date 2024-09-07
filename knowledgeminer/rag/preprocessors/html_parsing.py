from llama_index.core.node_parser import HTMLNodeParser

def parse_html_text_into_nodes(html_text, tags=["p", "h1"]):
    parser = HTMLNodeParser(tags=tags)  # optional list of tags
    nodes = parser.get_nodes_from_documents([html_text])
    return nodes
