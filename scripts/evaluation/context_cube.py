#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# Text normalization & metrics
# ------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: List[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def token_f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 used in SQuAD-style eval (on normalized tokens)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_f1_across_golds(prediction: str, gold_list: List[str]) -> float:
    return max((token_f1_score(prediction, g) for g in gold_list), default=0.0)


# ------------------------------
# Prompting
# ------------------------------
BASE_INSTRUCTION = (
    "You are a precise answerer. Always answer using a span copied verbatim from the provided information when possible. "
    "Never add extra words or explanations."
)

def build_final_prompt(context: str, query: str) -> str:
    # Minimal revision to improve EM/F1: encourage verbatim span answers.
    final_prompt = f"""
{BASE_INSTRUCTION}
Based on all the information gathered: {context}
provide a final answer to the original query: "{query}".
You must directly output the final answer without any other explanations.
If the query asks the dates or locations, only output the specific dates and locations.
If the answer is a city, just output the city name and do not output the country it belongs to.
If it is a yes-or-no query, only output yes or no.
If the query asks who, only output the person name.
If the query asks the comparison between two things, only output the one you think is correct without any other explanations.
If the query asks the nationality of someone, directly output the country name, e.g., Denmark or France. Do not output Danish or French.

Important: You MUST directly answer the question without any other text and thinking.
query: {query}
Your Answer:
"""
    return final_prompt.strip()


# ------------------------------
# Model client
# ------------------------------
def llm_chat_complete(
    api_url: str,
    model: str,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 20,
    top_p: float = 1.0,
    retries: int = 3,
    timeout: int = 120,
) -> str:
    """Call OpenAI-compatible Chat Completions API and return text."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": BASE_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        # "max_tokens": max_tokens,
        "top_p": top_p,
        "n": 1,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(body),
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                # Strip surrounding quotes/backticks/periods that sometimes slip in
                text = text.strip("`\"' \n\t")
                return text
            else:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_err = e
        # backoff
        sleep_s = min(2 ** attempt, 8)
        time.sleep(sleep_s)
    raise last_err if last_err else RuntimeError("Unknown API failure")


# ------------------------------
# Data loading helpers
# ------------------------------
def load_context_map(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect: {question: {"golden_answers": [...], "context_with_info": "..."}}
    return data


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "dataset_file must be a JSON list"
    return data


def synth_context_from_dataset_item(item: Dict[str, Any]) -> str:
    """Fallback: builds a readable 'context' string from dataset['context'] format if available."""
    if "context" not in item:
        return ""
    chunks = []
    for title, paras in item["context"]:
        text = "\\n".join(paras) if isinstance(paras, list) else str(paras)
        chunks.append(f"Doc (Title: {title})\\n{text}")
    return "\\n\\n".join(chunks)


# ------------------------------
# Main pipeline
# ------------------------------
def run(
    context_file: str,
    dataset_file: str,
    output_file: str,
    api_url: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> None:
    q2ctx = load_context_map(context_file)
    dataset = load_dataset(dataset_file)

    results = []
    for idx, item in tqdm(enumerate(dataset, start=1)):
        question = item.get("question", "").strip()
        gold_from_dataset = item.get("answer", "").strip()
        if question not in q2ctx:
            continue
        ctx_entry = q2ctx.get(question, {})
        golden_list: List[str] = []
        if gold_from_dataset:
            golden_list.append(gold_from_dataset)
        if ctx_entry and isinstance(ctx_entry.get("golden_answers"), list):
            golden_list.extend([str(x) for x in ctx_entry["golden_answers"] if isinstance(x, str)])
        # Ensure at least one gold exists to avoid division by zero in F1 logic downstream
        if not golden_list:
            golden_list = [gold_from_dataset] if gold_from_dataset else []

        # Determine context
        context_text = ctx_entry.get("context_with_info")
        if not context_text:
            context_text = synth_context_from_dataset_item(item)

        # Build prompt & query model
        prompt = build_final_prompt(context_text, question)
        try:
            pred = llm_chat_complete(
                api_url=api_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            pred = ""  # fail-safe
            print(f"[WARN] API failure on index {idx}: {e}")

        # Compute metrics
        em = em_check(pred, golden_list) if golden_list else 0
        f1 = best_f1_across_golds(pred, golden_list) if golden_list else 0.0

        # Compose output record
        out = {
            "index": idx,
            "question": question,
            "gold_answer": golden_list[0] if golden_list else "",
            "predicted_answer": pred,
            "em_score": float(em),
            "f1_score": float(f1),
        }
        results.append(out)
        
    avg_em = sum(result['em_score'] for result in results) / len(results)
    avg_f1 = sum(result['f1_score'] for result in results) / len(results)
    print(f"Average EM: {avg_em}, Average F1: {avg_f1}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Question Answering with Context (vLLM OpenAI-compatible)")
    parser.add_argument("--context_file", type=str, required=True, help="JSON file: {question: {golden_answers, context_with_info}}")
    parser.add_argument("--dataset_file", type=str, required=True, help="JSON list dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the output JSON")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", help="Base URL for OpenAI-compatible API server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen/Qwen2.5-7B-Instruct", help="Model name configured on the server")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max new tokens for generation")
    args = parser.parse_args()

    run(
        context_file=args.context_file,
        dataset_file=args.dataset_file,
        output_file=args.output_file,
        api_url=args.api_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
