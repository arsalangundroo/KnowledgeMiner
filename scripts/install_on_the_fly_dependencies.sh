#conda create -n "know_miner" python==3.11

conda activate know_miner_new
pip3 install llama-index
pip3 install llama-index-vector-stores-chroma
pip3 install llama-index-embeddings-azure-openai
pip3 install llama-index-llms-azure-openai
pip3 install llama-index-vector-stores-faiss
#pip3 uninstall openai
#pip3 install googletrans==4.0.0-rc1
pip3 install openai
pip3 install ragas
pip3 install llama-index-postprocessor-colbert-rerank
pip3 install FlagReranker
pip install llama-index-postprocessor-flag-embedding-reranker
pip3 install FlagReranker install git+https://github.com/FlagOpen/FlagEmbedding.git
pip3 install llama-index-retrievers-bm25


