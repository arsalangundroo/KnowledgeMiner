from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name)
    return embed_model