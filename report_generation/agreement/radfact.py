import hashlib
import json
import os
import threading
import time
import traceback

import httpx
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from tqdm import tqdm
from typing import Dict, List, Optional
NUM_CALLS = 0
NUM_TOKENS = 0
def make_prompt(ref_finding: str,pred_finding:str):
    text = ''
    #text = 'You are a medical report evaluator. For the CLAIM below, determine if it is logically entailed by (supported by/consistent with) the REFERENCE.\n\n'
    #text += 'Answer with only "ENTAILMENT", "CONTRADICTION", or "NEUTRAL".\n\n'
    text += 'REFERENCE:\n'
    text += ref_finding
    text +='\n\n'
    text += 'CLAIMS:\n'
    text += pred_finding
    return text

def compute_radfact(
    predictions: List[str],
    references: List[str],
    radfact_model: str,
    cache_file: Optional[str] = None,
    ollama_url: Optional[str] = None,
    max_workers: int = 1,
    rate_limit_rpm: int = 400,
    temperature:float = 0.0
) -> Dict[str, float]:
    """
    Compute RadFact score. Support Ollama and OpenAI.
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        radfact_model: Model to use for entailment verification (default: gpt-5-nano)
        cache_file: Path to cache file for storing results (optional)
        ollama_url: URL for Ollama API (required for local models like llama70b)
        max_workers: Number of parallel workers (default: 10, safe for 500 RPM limit)
    
    Returns:
        Dictionary with:
            - 'radfact-precision': Logical precision
            - 'radfact-recall': Logical recall
            - 'radfact-f1': Logical F1 score
    """
    try:
        return _compute_radfact(
            predictions=predictions,
            references=references,
            radfact_model=radfact_model,
            cache_file=cache_file,
            ollama_url=ollama_url,
            max_workers=max_workers,
            rate_limit_rpm=rate_limit_rpm,
            temperature=temperature
        )
    except Exception as e:
        print(f"Warning: RadFact computation failed: {e}")
        traceback.print_exc()
        return {
            'radfact-precision': 0.0,
            'radfact-recall': 0.0,
            'radfact-f1': 0.0
        }

def _compute_radfact(
    predictions: List[str],
    references: List[str],
    radfact_model: str = 'gpt-5-nano',
    cache_file: Optional[str] = None,
    ollama_url: Optional[str] = None,
    max_workers: int = 5,
    rate_limit_rpm: int = 400,
    temperature:float = 0.0
) -> Dict[str, float]:
    """
    see _compute_radfact for implementation details.
    """        
    cache = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            if cache and isinstance(list(cache.values())[0], dict):
                print(f"Loaded {len(cache)} cached sample results from {cache_file}")
            else:
                raise ValueError("Cazzi")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    def get_sample_cache_key(pred: str, ref: str) -> str:
        """Generate cache key for a sample (prediction-reference pair)."""
        content = f"{pred}|{ref}|{radfact_model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_cache():
        """Save cache to file."""
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    # No Ollama url => OpenAI
    use_ollama = ollama_url is not None
    
    if use_ollama:
        # Ollama setup
        print(f"Using Ollama model '{radfact_model}' at {ollama_url}, temp = {temperature}",flush=True)
        client = "ollama"
    else:
        # OpenAI setup
        assert os.getenv('OPENAI_API_KEY'), "OPENAI_API_KEY missing"
        try:
            http_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True
            )
            client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                http_client=http_client
            )
        except TypeError as e:
            client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                max_retries=3,
                timeout=60.0
            )
    
    # Helper function to make API call with rate limiting
    # OpenAI has some rate_limit_rpm (check the dashboard) that changes based on account
    # level and the model used. Check them.
    api_call_times = []
    api_lock = threading.Lock()
    
    def make_api_call(prompt: str, max_retries: int = 5) -> str:
        """Make a single API call with rate limiting and retry logic."""
        global NUM_CALLS
        NUM_CALLS +=1

        global NUM_TOKENS
        #print('made api call')
        '''if NUM_CALLS % 100 == 0:
            print(f'Already made {NUM_CALLS} API calls, tokens: {NUM_TOKENS}')'''
        with api_lock:
            current_time = time.time()
            # Remove calls older than 60 seconds
            api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
            
            # If we're at the limit, wait until we can make another call
            if len(api_call_times) >= rate_limit_rpm:
                oldest_call = api_call_times[0]
                wait_time = 60 - (current_time - oldest_call)
                if wait_time > 0:
                    time.sleep(wait_time + 0.5)
                    current_time = time.time()
                    api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
            
            api_call_times.append(current_time)
        
        # API call
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                if client == "ollama":
                    data = {
                        "model": radfact_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": temperature
                        }
                    }
                    resp = requests.post(
                        f"{ollama_url}/api/chat",
                        json=data,
                        timeout=120
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    content = result['message']['content'].strip()
                    prompt_tokens = result["prompt_eval_count"]
                    completion_tokens = result["eval_count"]
                    total_tokens = prompt_tokens + completion_tokens
                    NUM_TOKENS+=total_tokens
                    if not content:
                        raise ValueError(f"Empty response from Ollama API (model: {radfact_model})")

                    return content
                else:
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable not set")
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    is_gpt5_plus = "gpt-5" in radfact_model.lower()
                    token_param = "max_completion_tokens" if is_gpt5_plus else "max_tokens"
                    data = {
                        "model": radfact_model,
                        "messages": [{"role": "user", "content": prompt}],
                        token_param: 4096
                    }
                    if not is_gpt5_plus:
                        data["temperature"] = 0.0
                    
                    resp = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=1000
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    if 'error' in result:
                        raise ValueError(f"API error: {result['error']}")
                    
                    if 'choices' not in result or not result['choices']:
                        raise ValueError(f"No choices in API response: {result}")
                    
                    content = result['choices'][0]['message']['content'].strip()
                    
                    if not content:
                        raise ValueError(f"Empty response from OpenAI API (model: {radfact_model})")
                    
                    return content
                    
            except requests.exceptions.Timeout as e:
                last_error = TimeoutError(f"API request timed out: {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                    tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Timeout error, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue
                
            except requests.exceptions.HTTPError as e:
                # Handle rate limiting (429) with exponential backoff
                if e.response.status_code == 429:
                    retry_count += 1
                    if retry_count <= max_retries:
                        # Check if API provides retry-after header
                        retry_after = e.response.headers.get('retry-after')
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except:
                                wait_time = min(2 ** retry_count * 5, 120)  # Exponential backoff
                        else:
                            wait_time = min(2 ** retry_count * 5, 120)  # Exponential backoff, max 120s
                        
                        tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Rate limit hit (429), waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries exceeded for 429
                        try:
                            error_detail = e.response.json()
                            last_error = ValueError(f"HTTP 429 (Rate limit exceeded after {max_retries} retries): {error_detail}")
                        except:
                            last_error = ValueError(f"HTTP 429 (Rate limit exceeded after {max_retries} retries): {e}")
                        break
                else:
                    # Other HTTP errors - try to extract error message
                    try:
                        error_detail = e.response.json()
                        last_error = ValueError(f"HTTP {e.response.status_code}: {error_detail}")
                    except:
                        last_error = ValueError(f"HTTP {e.response.status_code}: {e}")
                    
                    # Retry on 5xx errors
                    if 500 <= e.response.status_code < 600:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = min(2 ** retry_count, 60)
                            tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Server error ({e.response.status_code}), waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                    break
                    
            except requests.exceptions.RequestException as e:
                last_error = ConnectionError(f"API request failed: {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = min(2 ** retry_count, 60)
                    tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Connection error, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("API call failed after retries with unknown error")

    def process_sample(idx, pred_text, ref_text):
        """Process one sample (prediction-reference pair) using 1-vs-1 comparisons."""
        try:
            # ----- Cache Check -----
            sample_key = get_sample_cache_key(pred_text, ref_text)
            if sample_key in cache:
                cached = cache[sample_key]
                return idx, cached["precision"], cached["recall"], None

            # ----- Split Findings -----
            pred_findings = [
                line.strip() for line in pred_text.strip().split("\n") if line.strip()
            ]
            ref_findings = [
                line.strip() for line in ref_text.strip().split("\n") if line.strip()
            ]
            if not ref_findings:
                print('empty ref')
                return idx, None, None, "empty_ref"

            # ============================================================
            # PRECISION (1 predicted vs each reference finding)
            # ============================================================

            entailed_count = 0

            for pred_idx, pred_finding in enumerate(pred_findings):
                is_entailed = False

                for ref_finding in ref_findings[pred_idx:] + ref_findings[:pred_idx]:
                    precision_prompt = make_prompt(ref_finding=ref_finding,pred_finding=pred_finding)
                    try:
                        response = make_api_call(precision_prompt)
                        if response and "ENTAILMENT" in response.upper():
                            is_entailed = True
                            break  # stop checking other reference findings

                    except Exception as e:
                        tqdm.write(
                            f"    [ERROR] Precision API call failed (sample {idx}): {type(e).__name__}: {e}"
                        )

                if is_entailed:
                    entailed_count += 1

            precision = (
                entailed_count / len(pred_findings)
                if pred_findings else 0.0
            )

            # ============================================================
            # RECALL (1 reference vs each predicted finding)
            # ============================================================

            covered_count = 0

            for ref_idx, ref_finding in enumerate(ref_findings):
                is_covered = False

                for pred_finding in pred_findings[ref_idx:] + pred_findings[:ref_idx]:
                    recall_prompt = make_prompt(ref_finding=pred_finding,pred_finding=ref_finding)
                    try:
                        response = make_api_call(recall_prompt)
                        if response and "ENTAILMENT" in response.upper():
                            is_covered = True
                            break  # stop checking other predicted findings
                    except Exception as e:
                        tqdm.write(
                            f"    [ERROR] Recall API call failed (sample {idx}): {type(e).__name__}: {e}"
                        )

                if is_covered:
                    covered_count += 1

            recall = (
                covered_count / len(ref_findings)
                if ref_findings else 0.0
            )

            # ----- Cache Result -----
            cache[sample_key] = {
                "precision": precision,
                "recall": recall
            }

            return idx, precision, recall, None

        except Exception as e:
            error_msg = (
                f"Sample {idx} processing failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            tqdm.write(f"    [ERROR] {error_msg}")
            return idx, None, None, error_msg

    # Thread-safe cache saving
    cache_lock = threading.Lock()
    
    def save_cache_safe():
        with cache_lock:
            save_cache()
    
    all_precisions = []
    all_recalls = []
    failed_samples = 0
    results = {}  # Store results with original indices
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, idx, pred, ref): idx
            for idx, (pred, ref) in enumerate(zip(predictions, references))
        }
        
        # Collect results as they complete with progress bar
        for future in tqdm(as_completed(futures), 
                            total=len(predictions),
                            desc="  RadFact evaluation",
                            unit="reports"):
            idx, precision, recall, error = future.result()
            
            if error:
                if error != "empty_ref":
                    tqdm.write(f"    Warning: Failed to process sample {idx}: {error}")
                    failed_samples += 1
            else:
                results[idx] = (precision, recall)
                # Save cache periodically (every 10 samples)
                if len(results) % 10 == 0:
                    save_cache_safe()
    
    # Final cache save
    save_cache_safe()
    
    # Sort results by original index
    for idx in sorted(results.keys()):
        precision, recall = results[idx]
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    if failed_samples > 0:
        print(f"Warning: {failed_samples} samples failed during RadFact computation")
    
    # Compute average precision and recall
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    
    # Compute F1
    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0.0
    
    print(f"RadFact computed from {len(all_precisions)} samples:")
    print(f"\tPrecision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {f1:.4f}")
    
    return {
        'radfact-precision': avg_precision,
        'radfact-recall': avg_recall,
        'radfact-f1': f1
    }