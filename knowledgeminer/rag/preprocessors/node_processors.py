from typing import List

from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.schema import BaseNode
from llama_index.schema import IndexNode



class DocumentsToNodesProcessor(object):

    #TODO: replace this with a generic tranformation pipeline that takes in a config for transformations to be applied.
    @staticmethod
    def docs_to_nodes_sent_chunk_with_title_extraction(documents, chunk_size=1024, chunk_overlap=64, llm=None):
        # create the pipeline with transformations
        try:
            #TODO: to make this generic, read from configs what other transformations need to be applied after chunking
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=chunk_size, chunk_overlap= chunk_overlap),
                    TitleExtractor(llm),
                    #embed_model
                ]
            )
            # run the pipeline
            base_nodes = pipeline.run(documents=documents)
            return base_nodes
        except Exception as e:
            print(e)

    @staticmethod
    def create_index_nodes(base_nodes: List[BaseNode], sub_chunk_sizes:List[int],
                           chunk_overlap=20, Splitter = SentenceSplitter)->tuple[List[IndexNode],dict[str,IndexNode]]:
        try:
            sub_node_parsers = [
                Splitter(chunk_size=c, chunk_overlap=chunk_overlap) for c in sub_chunk_sizes]
            all_nodes = []
            for base_node in base_nodes:
                for n in sub_node_parsers:
                    sub_nodes = n.get_nodes_from_documents([base_node])
                    sub_inodes = [
                        IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                    ]
                    all_nodes.extend(sub_inodes)

                # also add original node to node
                original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                all_nodes.append(original_node)
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            return all_nodes, all_nodes_dict
        except Exception as e:
            print(e)








