from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

def create_hf_embed_model(model_name="BAAI/bge-small-en-v1.5", use_fp16=False):
    if use_fp16:
        embed_model = HuggingFaceEmbedding(model_name,model_kwargs={"torch_dtype": torch.float16})
    else:
        embed_model = HuggingFaceEmbedding(model_name)
    return embed_model