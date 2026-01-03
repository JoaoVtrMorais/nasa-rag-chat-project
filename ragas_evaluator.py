import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List
from dotenv import load_dotenv

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


load_dotenv()


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    openai_key = os.getenv("OPEN_AI_KEY")

    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    if openai_key.startswith("voc"):
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1"
        ))
    else:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_key,
        ))

    if openai_key.startswith("voc"):
        langchain_openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
            openai_api_base="https://openai.vocareum.com/v1"
        )
    else:
        langchain_openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )
      
    # TODO: Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)
    
    # TODO: Define an instance for each metric to evaluate
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm),
        NonLLMContextPrecisionWithReference(),
        RougeScore(),
        BleuScore()
    ]
    # TODO: Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        contexts=contexts
    )

    results = evaluate(
        samples=[sample],
        metrics=metrics,
        embeddings=evaluator_embeddings 
    )

    # TODO: Return the evaluation results
    return {
        metric: float(results[metric][0])
        for metric in results
    }
