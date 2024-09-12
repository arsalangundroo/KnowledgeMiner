#conda create -n "know_miner" python==3.11

conda activate know_miner
pip3 install llama-index-vector-stores-chroma
pip3 install llama-index-embeddings-azure-openai
pip3 install llama-index-llms-azure-openai
pip3 install llama-index-vector-stores-faiss
pip3 uninstall openai
pip3 install googletrans==4.0.0-rc1
pip3 install openai