from llama_index.core import Settings
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform, HyDEQueryTransform,
)
from llama_index.core.query_engine import MultiStepQueryEngine, TransformQueryEngine


def get_multi_step_query_transform_engine(query_engine):
    #TODO: Check wherer/how to correctly pass response_synthesizer and re-ranker
    step_decompose_transform = StepDecomposeQueryTransform(Settings.llm, verbose=True)

    query_engine = MultiStepQueryEngine(
        query_engine, query_transform=step_decompose_transform
    )
    return query_engine


def get_hyde_query_transform_engine(query_engine):
    hyde = HyDEQueryTransform(include_original=True)
    query_engine = TransformQueryEngine(query_engine, hyde)
    return query_engine
