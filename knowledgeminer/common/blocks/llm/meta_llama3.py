from transformers import AutoTokenizer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

#hf_token = "hf_"

def create_hf_llama_3_1(model_name="meta-llama/Meta-Llama-3-8B-Instruct",tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #token=hf_token,
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Optional quantization to 4bit
    # import torch
    # from transformers import BitsAndBytesConfig

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    llm = HuggingFaceLLM(
        model_name=model_name,
        model_kwargs={
            #"token": hf_token,
            "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
            # "quantization_config": quantization_config
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        tokenizer_name=tokenizer_name,
        #tokenizer_kwargs={"token": hf_token},
        stopping_ids=stopping_ids,
    )


    ## You can deploy the model on HF Inference Endpoint and use it

    # from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

    # llm = HuggingFaceInferenceAPI(
    #     model_name="",
    #     token=''
    # )

    return llm