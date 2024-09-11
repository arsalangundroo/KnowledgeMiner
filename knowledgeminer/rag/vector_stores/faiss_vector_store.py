from typing import Any

import faiss
import llama_index.embeddings
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class FaissLamaIndexClient(object):
    def __init__(self, embed_model, dim, llm: Any = "default",
                 faiss_index_type: faiss.Index = faiss.IndexFlatL2):
        self._dim = dim  # dimensions of text-ada-embedding-002 is 1536
        self._faiss_index = faiss_index_type(self._dim)
        self._embed_model = embed_model
        self._vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        #self._service_context = ServiceContext.from_defaults(embed_model=self._embed_model, llm=llm)
        self._index = None


    def create_vector_store_index(self, input_nodes):
        logging.info("Creating Faiss Vector Store Index ............")
        self._index = VectorStoreIndex(input_nodes,
                                       storage_context=self._storage_context, show_progress=True)
        logging.info(0, "Vector Store Creation Complete!")
        return self._index

    def get_vector_store_index(self):
        return self._index

    # TODO: define getters for all attributes

    def add_documents_to_vector_store(self, nodes):
        raise NotImplementedError

    def save_to_persistent_storage(self, url: str):
        self._index.storage_context.persist(persist_dir=url)
        logging.info("Saved Faiss index successfully!!!")

    @staticmethod
    def load_from_persistent_storage(url: str, embed_model=None, dim=1536):
        try:
            vector_store = FaissVectorStore.from_persist_dir(url)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=url
            )
            loaded_index = load_index_from_storage(storage_context=storage_context)

            loaded_faiss_vector_store_client = FaissLamaIndexClient(
                embed_model,
                dim=vector_store.client.d
            )
            loaded_faiss_vector_store_client._index=loaded_index
            return loaded_faiss_vector_store_client

        except Exception as e:
            logging.exception(e)
