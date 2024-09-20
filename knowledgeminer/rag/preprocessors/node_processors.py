from typing import List

from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.schema import BaseNode
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
)



class DocumentsToNodesProcessor(object):

    #TODO: replace this with a generic tranformation pipeline that takes in a config for transformations to be applied.
    @staticmethod
    def docs_to_nodes_sent_chunk_with_title_extraction(documents, chunk_size=1024, chunk_overlap=64, llm=None):
        # create the pipeline with transformations
        try:
            #TODO: to make this generic, read from configs what other transformations need to be applied after chunking
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                    #TitleExtractor(llm),
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
                           sub_chunk_overlaps, Splitter = SentenceSplitter)->tuple[List[IndexNode],dict[str,IndexNode]]:
        try:
            sub_node_parsers = [
                Splitter(chunk_size=c, chunk_overlap=chunk_overlap) for c,chunk_overlap in zip(sub_chunk_sizes,sub_chunk_overlaps)]
            all_nodes = []
            for base_node in base_nodes:
                for n in sub_node_parsers:
                    sub_nodes = n.get_nodes_from_documents([base_node])
                    sub_nodes = [
                        IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                    ]
                    all_nodes.extend(sub_nodes)

                # also add original node to node
                original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                all_nodes.append(original_node)
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            return all_nodes, all_nodes_dict
        except Exception as e:
            print(e)

    @staticmethod
    def create_single_sentence_nodes_with_metadata_window(documents, window_size=3):
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes


    # TODO: Replace the below method and first method with a single generic method that can be configured through function arguments
    @staticmethod
    def create_index_nodes_from_sentence_window_chunking_for_recursive_retrieval(base_nodes: List[BaseNode],window_size=3) -> tuple[
        List[IndexNode], dict[str, IndexNode]]:
        try:
            node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            all_nodes = []
            for base_node in base_nodes:

                sub_nodes = node_parser.get_nodes_from_documents([base_node])
                sub_nodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_nodes)

                # also add original node to node
                original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                all_nodes.append(original_node)
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            return all_nodes, all_nodes_dict
        except Exception as e:
            print(e)






