import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from dotenv import load_dotenv

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        # NonLLMContextPrecisionWithReference,
        context_precision,
        BleuScore,
        RougeScore,
    )
    from ragas import evaluate

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


load_dotenv()


def evaluate_response_quality(
    question: str, answer: str, contexts: List[str], reference: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not question.strip():
        return {"error": "Question is empty or missing"}
    if not answer or not answer.strip():
        return {"error": "Answer is empty or missing"}
    if not contexts or not any(context.strip() for context in contexts):
        return {"error": "Contexts are empty or missing"}

    openai_key = os.getenv("OPEN_AI_KEY")
    if not openai_key:
        return {"error": "OPEN_AI_KEY not found in environment variables"}

    # Create evaluator LLM with model gpt-3.5-turbo
    if openai_key.startswith("voc"):
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_key,
                base_url="https://openai.vocareum.com/v1",
            )
        )
    else:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_key,
            )
        )

    if openai_key.startswith("voc"):
        langchain_openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
            openai_api_base="https://openai.vocareum.com/v1",
        )
    else:
        langchain_openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )

    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)

    # Define an instance for each metric to evaluate
    metrics = [Faithfulness(), ResponseRelevancy()]

    sample_kwargs = {
        "user_input": question.strip(),
        "response": answer.strip(),
        "retrieved_contexts": [ctx.strip() for ctx in contexts if ctx.strip()],
    }

    if reference and reference.strip():
        sample_kwargs["reference"] = reference.strip()
        metrics.extend(
            [
                RougeScore(), 
                BleuScore(), 
                # NonLLMContextPrecisionWithReference() # Requires 'reference_contexts' (list of ideal chunks), incompatible with ROUGE/BLEU which use 'reference' (ideal answer string).
                context_precision,
            ]
        )

    # Evaluate the response using the metrics
    sample = SingleTurnSample(**sample_kwargs)

    dataset = EvaluationDataset(samples=[sample])

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Return the evaluation results
    if hasattr(results, "scores") and len(results.scores) > 0:
        evaluation_results = {
            key: float(value) for key, value in results.scores[0].items()
        }
    else:
        return {"error": "No scores returned by RAGAS evaluation"}

    return evaluation_results


def load_evaluation_dataset(
    file_path: str = "evaluation_dataset.txt",
) -> List[Dict[str, str]]:
    """Load questions and ground truth references from evaluation_dataset.txt"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                raise ValueError(
                    f"Line {line_num}: Expected format 'question | reference'"
                )
            question, reference = [part.strip() for part in line.split("|", 1)]
            samples.append({"question": question, "reference": reference})

    if len(samples) < 5:
        raise ValueError("Evaluation dataset must contain at least 5 questions")
    return samples


def run_batch_evaluation(model_outputs: List[Dict[str, str]]) -> Dict:
    """
    Runs evaluation over the full test set using evaluation_dataset.txt
    model_outputs: list of dicts with keys 'answer' and 'contexts' (in same order as dataset)
    Returns individual results + aggregate statistics
    """
    try:
        test_set = load_evaluation_dataset()
    except Exception as e:
        return {"error": str(e)}

    if len(model_outputs) != len(test_set):
        return {"error": f"Expected {len(test_set)} answers, got {len(model_outputs)}"}

    individual_results = []
    all_scores = {}

    for i, item in enumerate(model_outputs):
        q_data = test_set[i]
        result = evaluate_response_quality(
            question=q_data["question"],
            answer=item.get("answer", ""),
            contexts=item.get("contexts", []),
            reference=q_data["reference"],
        )
        full_result = {
            "question": q_data["question"],
            "reference": q_data["reference"],
            "answer": item.get("answer", ""),
            "metrics": result if "error" not in result else {"error": result["error"]},
        }
        individual_results.append(full_result)

        # Collect valid scores for aggregate
        if "error" not in result:
            for metric, score in result.items():
                all_scores.setdefault(metric, []).append(score)

    # Compute aggregate statistics
    aggregate = {
        metric: {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
        for metric, values in all_scores.items()
    }

    return {
        "total_questions": len(test_set),
        "individual_results": individual_results,
        "aggregate_metrics": aggregate,
    }


if __name__ == "__main__":

    print("=== RAGAS EVALUATION TEST - APOLLO 13 ===\n")

    try:
        # Load the evaluation dataset
        test_set = load_evaluation_dataset("evaluation_dataset.txt")
        print(f"Successfully loaded dataset: {len(test_set)} questions\n")

        # Simulated outputs from your RAG system (replace this with your real pipeline later)
        model_outputs = [
            # 1. Call signs - GOOD answer
            {
                "answer": "When both vehicles were manned, the Command Module was called Odyssey and the Lunar Module Aquarius.",
                "contexts": [
                    "When both vehicles are manned, the call sign will be Odyssey for the CSM and Aquarius for the LM.",
                    "Voice calls during this mission were assigned in accordance with the following station operating procedures.",
                ],
            },
            # 2. Crew members - GOOD answer
            {
                "answer": "The crew consisted of Commander Jim Lovell, Command Module Pilot Jack Swigert, and Lunar Module Pilot Fred Haise.",
                "contexts": [
                    "CDR James A. (Jim) Lovell, Jr.",
                    "CMP John L. Swigert, Jr.",
                    "LMP Fred W. Haise, Jr.",
                ],
            },
            # 3. S-II ignition - GOOD answer
            {
                "answer": "The crew reported S-II ignition at GET 000:02:48.",
                "contexts": ["000:02:48 CDR S-II ignition."],
            },
            # 4. S-IVB cutoff - GOOD answer
            {
                "answer": "Houston informed the crew that the predicted S-IVB cutoff time was 12 plus 34, which is GET 00:12:34.",
                "contexts": ["Predicted cut-off on the S-IVB is 12 plus 34."],
            },
            # 5. O2 FLOW HIGH - MEDIUM answer (a bit vague)
            {
                "answer": "The O2 flow was high for a while. Houston said it was okay.",
                "contexts": [
                    "At 2 hours and 12 minutes, the 02 FLOW HIGH light came on...",
                    "Houston: it's nominal with the WASTE TANK VENT open... no sweat.",
                ],
            },
            # 6. LM jettison - GOOD answer
            {
                "answer": "LM jettison occurred around 141:30 GET.",
                "contexts": ["141:30:05 CC Okay, copy that. Farewell, Aquarius..."],
            },
            # 7. Recovery ship - BAD answer (hallucination)
            {
                "answer": "The recovery ship was the USS Ticonderoga in the Atlantic Ocean.",
                "contexts": [
                    "the Iwo Jima will be at the touchdown point",
                    "Mid-Pacific landing area",
                ],
            },
            # 8. Moonset GET - GOOD answer
            {
                "answer": "The final entry pad indicated moonset and Moon-check attitude at GET 142:38:19 with 178 degrees.",
                "contexts": ["142:38:19, 178;"],
            },
        ]

        print("Running batch evaluation with simulated answers...\n")

        # Run the full batch evaluation
        results = run_batch_evaluation(model_outputs)

        if "error" in results:
            print("Evaluation ERROR:", results["error"])
        else:
            print("=== INDIVIDUAL RESULTS ===\n")
            for i, res in enumerate(results["individual_results"], 1):
                print(f"Question {i}:")
                print(f"   Question: {res['question'][:80]}...")
                print(f"   Metrics: {res['metrics']}")
                print()

            print("=== OVERALL SUMMARY (AVERAGES) ===\n")
            agg = results["aggregate_metrics"]
            for metric, stats in agg.items():
                print(
                    f"{metric.replace('_', ' ').title():25} → Average: {stats['mean']:.3f} "
                    f"(min: {stats['min']:.3f}, max: {stats['max']:.3f})"
                )

            print(f"\nEvaluation completed on {results['total_questions']} questions.")

    except Exception as e:
        print("Error running the test:", str(e))
        import traceback

        traceback.print_exc()
