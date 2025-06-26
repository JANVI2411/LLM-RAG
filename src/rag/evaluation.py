import pandas as pd
import mlflow
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset, load_dataset

hf_token = ""

def generate_ragas_dataset(dataset_path: str):
    hf_dataset = load_dataset(dataset_path, token=hf_token)

    ragas_dataset = []
    for item in hf_dataset:
        query = item["question"]
        response = rag(query)
        source_documents = response["source_documents"]
        answer = response["answer"]
        ragas_dataset.append(
            {
                "question": item["question"],
                "ground_truth": item["answer"],
                "context": [item["context"]],
                "answer":answer,
                "retrieved_contexts":[d.page_content for d in source_documents]
            }
        )

    ragas_df = pd.DataFrame(ragas_dataset)
    ragas_df.to_csv("rag_evaluated_dataset.csv", index=False)
    # ragas_dataset_hf = Dataset.from_list(ragas_dataset)

def run_evaluation_pipeline(dataset_path: str, run_name: str = "RAG_Eval_Run"):
    df = pd.read_csv(dataset_path)
    ragas_dataset = Dataset.from_pandas(df)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset", dataset_path)
        mlflow.log_param("num_samples", len(df))

        results = evaluate(ragas_dataset, metrics=metrics)

        for metric in metrics:
            mlflow.log_metric(metric.name, results[metric.name])

        pd.DataFrame([results]).to_csv("results.csv", index=False)
        mlflow.log_artifact("results.csv")
        print("Evaluation logged to MLflow.")

def run_evaluation(dataset_path: str):
    df = pd.read_csv(dataset_path)
    ragas_dataset = Dataset.from_pandas(df)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(ragas_dataset, metrics=metrics)

    pd.DataFrame([results]).to_csv("results.csv", index=False)
    print("Evaluation csv saved to results.csv.")
