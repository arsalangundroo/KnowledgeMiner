import os

from llama_index.core import Settings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics.critique import harmfulness
from ragas import evaluate, RunConfig
import json
from datasets import Dataset
from tqdm import tqdm

from knowledgeminer.common.blocks.embeddings.azure_openai_for_langchain import \
    create_basic_azure_openai_embedding_client
from knowledgeminer.common.blocks.llm.azure_openai_for_langchain import create_basic_azure_openai_client

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]


def get_context_list_from_retrieved_nodes(source_nodes):
    contexts = []
    for node in source_nodes:
        contexts.append(node.text)
    return contexts


def create_evaluation_dataset_with_ground_truth_for_RAGAS(ground_truth_input_file, query_engine_pipeline, out_file="../out/ragas_eval_dataset.json"):
    eval_data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    with open(ground_truth_input_file,"r") as fp:
        input_json_list = json.load(fp)
    print(f"Total num. of evaluation questions: {len(input_json_list)}")
    print("Generating Evaluation Datset for RAGAS .............")
    for sample in tqdm(input_json_list):
        response = query_engine_pipeline.query(sample["Question"])
        eval_data_samples['question'].append(sample['Question'])
        eval_data_samples['answer'].append(response.response)
        eval_data_samples['contexts'].append(get_context_list_from_retrieved_nodes(response.source_nodes))
        eval_data_samples['ground_truth'].append(sample["Answer"])

    with open(out_file, 'w') as fp:
        json.dump(eval_data_samples, fp)
    print([len(x) for x in eval_data_samples['contexts'][:20]])
    dataset = Dataset.from_dict(eval_data_samples)
    return dataset


def run_evaluation_with_ground_truth_dataset(eval_dataset_with_gt,eval_dataset_json_file=None):
    if not eval_dataset_with_gt:
        #TODO: load and create eval_dataset from its json file
        with open(eval_dataset_json_file, 'r') as fp:
            json_str = fp.read()
        eval_data_samples = json.loads(json_str)
        eval_dataset_with_gt = Dataset.from_dict(eval_data_samples)
        print(f"Total num. of evaluation questions: {len(eval_data_samples['question'])}")
        print(len(eval_data_samples['contexts'][1]))
    print("Computing Evaluation Metrics with RAGAS ::::::::::: ")
    azure_embeddings = create_basic_azure_openai_embedding_client()

    result = evaluate(
        eval_dataset_with_gt,
        metrics=metrics,
        llm=create_basic_azure_openai_client(),
        embeddings=azure_embeddings,
        #raise_exceptions=True,
        run_config=RunConfig(timeout=250)
    )

    df = result.to_pandas()
    print(df.head())

    return df
