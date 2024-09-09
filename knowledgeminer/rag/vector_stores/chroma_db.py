import logging
from typing import Any

import llama_index.embeddings
from chromadb import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb


class ChromaDBLamaIndexClient(object):
    def __init__(self, collection_name: str, embed_model, persist_dir="./new_chroma_db_storage/", llm: Any = "default"):
        self._chroma_client = chromadb.PersistentClient(persist_dir)
        self._chroma_collection = self._chroma_client.create_collection(collection_name)
        #self._embed_model = embed_model
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        # self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        # self._service_context = ServiceContext.from_defaults(embed_model=self._embed_model, llm=llm)
        self._index = None

    # TODO: Define the following constructor using *args params for constructor overloading.

    def _create_client_from_persistent_index(self, client, collection, vector_store, loaded_index):

        self._chroma_client = client
        self._chroma_collection = collection
        #self._embed_model = service_context.embed_model
        self._vector_store = vector_store
        self._index = loaded_index
        return self

    def create_vector_store_index(self, input_nodes):
        print("Creating Vector Store Index ............")
        # self._index = VectorStoreIndex.from_documents(input_nodes, storage_context=self._storage_context,
        #                                                service_context=self._service_context, show_progress=True)
        # self._index = VectorStoreIndex(input_nodes, storage_context=self._storage_context,
        #                                service_context=self._service_context, show_progress=True)
        self._index = VectorStoreIndex(input_nodes, show_progress=True)
        print("Vector Store Creation Complete!")

        return self._index

    def get_vector_store_index(self):
        return self._index

    # TODO: define getters for all attributes

    def add_documents_to_vector_store(self, nodes):
        raise NotImplementedError

    def save_to_persistent_storage(self, url):
        self._index.storage_context.persist(persist_dir=url)
        print(f"Saved chroma_db index successfully at {url}!!!")

    @staticmethod
    def load_from_persistent_storage(url: str, collection_name: str):
        try:
            # chroma_db_client = chromadb.PersistentClient(path=url)
            # chroma_collection = chroma_db_client.get_or_create_collection(collection_name)
            # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # loaded_index = VectorStoreIndex.from_vector_store(
            #     vector_store,
            # )
            # loaded_chroma_db_vector_store_client = _create_client_from_persistent_index(
            #     client=chroma_db_client,
            #     collection=chroma_collection,
            #     vector_store=vector_store,
            #     loaded_index=loaded_index
            # )

            loaded_chroma_db_vector_store_client = ChromaDBLamaIndexClient(None,None)
            loaded_chroma_db_vector_store_client._chroma_client= chromadb.PersistentClient(path=url)
            loaded_chroma_db_vector_store_client._chroma_collection = loaded_chroma_db_vector_store_client._chroma_client.get_or_create_collection(collection_name)
            loaded_chroma_db_vector_store_client._vector_store = ChromaVectorStore(chroma_collection=loaded_chroma_db_vector_store_client._chroma_collection)
            loaded_chroma_db_vector_store_client._index = VectorStoreIndex.from_vector_store(
                loaded_chroma_db_vector_store_client._vector_store,
            )
            return loaded_chroma_db_vector_store_client

        except Exception as e:
            logging.exception(e)


