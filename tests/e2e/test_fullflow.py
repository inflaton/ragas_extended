import json
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

# import langchain

# langchain.debug = True


def load_baseline_dataset(use_ground_truths):
    ds = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
    if use_ground_truths:
        ds = ds.map(
            lambda record: {
                "answer": record["ground_truths"][0],
                "question": record["question"],
                "contexts": record["contexts"],
            },
            batched=False,
        )

    return ds


def test_evaluate_e2e_baseline_dataset():
    ds = load_baseline_dataset(True)
    baseline_index = [3, 9, 14, 16, 22, 24, 26, 27]
    result = evaluate(
        ds.select(baseline_index),
        metrics=[faithfulness, answer_relevancy],
    )
    print(result)

    filename = "test_evaluate_e2e_baseline_dataset.csv.log"
    print(f"Saving results to {filename} ...")

    result.to_pandas().to_csv(filename)

    file = open(filename, "a")  # append mode
    file.write(f"\n\n# Ragas overall scores: {result}\n")
    file.close()

    assert result["faithfulness"] > 0.9
