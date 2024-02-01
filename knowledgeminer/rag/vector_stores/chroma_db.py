from typing import Any

import llama_index.embeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb


class ChromaDBCollectionLamaIndexClient(object):
    def __init__(self, collection_name: str, embed_model: llama_index.embeddings.BaseEmbedding, llm: Any = "default"):
        self._chroma_client = chromadb.EphemeralClient()
        self._chroma_collection = self._chroma_client.create_collection(collection_name)
        self._embed_model = embed_model
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._service_context = ServiceContext.from_defaults(embed_model=self._embed_model, llm=llm)
        self._index = None

    def create_vector_store_index(self, input_nodes):
        print("Creating Vector Store Index ............")
        #self._index = VectorStoreIndex.from_documents(input_nodes, storage_context=self._storage_context,
        #                                                service_context=self._service_context, show_progress=True)
        self._index = VectorStoreIndex(input_nodes, storage_context=self._storage_context,
                                                      service_context=self._service_context, show_progress=True)
        print("Vector Store Creation Complete!")

        return self._index

    def get_vector_store_index(self):
        return self._index

    #TODO: define getters for all attributes

    def add_documents_to_vector_store(self,nodes):
        raise NotImplementedError

    def save_to_persistent_storage(self,url):
        self._index.storage_context.persist(persist_dir=url)
        print("Saved chroma_db index successfully!!!")

    @staticmethod
    def load_from_persistent_storage(self,url:str):
        raise NotImplementedError
