import argparse
import json
from datetime import datetime
from pathlib import Path


def load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_ragas(dataset: list[dict]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "ragas evaluation dependencies not installed. Install requirements first."
        ) from exc

    eval_dataset = Dataset.from_list(dataset)
    score = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return score.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for FraudShield.")
    parser.add_argument("--dataset", default="eval/test_questions.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--output-dir", default="eval/results", help="Directory for timestamped result files")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    results = run_ragas(dataset)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"ragas-results-{timestamp}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
