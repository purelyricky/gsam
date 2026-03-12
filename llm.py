"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""
import time
import random
import json
from datetime import datetime
import openai
from logger import log_llm_call, log_problematic_request

def timed_llm_call(client, api_provider, model, prompt, role, call_id, max_tokens=4096, log_dir=None,
                   sleep_seconds=15, retries_on_timeout=1000, attempt=1, use_json_mode=False):
    """
    Make a timed LLM call with error handling and retry logic.
    
    EMPTY RESPONSE HANDLING STRATEGY:
    - Empty responses (None content) and truncated responses (finish_reason=length) both
      return "INCORRECT_DUE_TO_EMPTY_RESPONSE" so callers can detect and EXCLUDE them from
      accuracy math entirely — they are neither correct nor incorrect.
    - Empty content is retried up to 3 times before giving up.
    - finish_reason=length is NOT retried (more tokens won't fix a truncated prompt).
    - All problematic requests are logged to problematic_requests/ for analysis.
    
    Args:
        client: API client
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging (generator, reflector, curator)
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        max_tokens: Maximum tokens to generate
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries
        retries_on_timeout: Maximum number of retries for timeouts/rate limits/empty responses
        attempt: Current attempt number (for recursive calls)
        use_json_mode: Whether to use JSON mode for structured output
    
    Returns:
        tuple: (response_text, call_info_dict)
        
    Special return values for empty/truncated responses:
        - Returns "INCORRECT_DUE_TO_EMPTY_RESPONSE, ..." (repeated 4 times for FiNER format)
        - Callers detect this string and EXCLUDE the sample from accuracy calculation entirely
    """
    start_time = time.time()
    prompt_time = time.time()
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    
    # Check if we're using API key mixer for dynamic key rotation on retries
    using_key_mixer = False
    
    while True:
        try:
            # Get client
            active_client = client

            # Prepare API call parameters
            if api_provider == "openai":
                max_tokens_key = "max_completion_tokens"
            else:
                max_tokens_key = "max_tokens"

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                max_tokens_key: max_tokens
            }
            
            # Add JSON mode if requested
            if use_json_mode:
                api_params["response_format"] = {"type": "json_object"}
            call_start = time.time()
            response = active_client.chat.completions.create(**api_params)
            call_end = time.time()
            
            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("Empty response from API")
            
            response_time = time.time()
            total_time = response_time - start_time
            response_content = response.choices[0].message.content
            
            if response_content is None:
                raise Exception("API returned None content")

            # Detect finish_reason == "length": response was cut off before a valid answer
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise Exception(f"Response truncated (finish_reason=length) — likely hit max_tokens={max_tokens}")

            # Detect vLLM/server error payloads embedded in the response content.
            # These look like {"object": "error", "message": "..."} or {"error": {...}}
            # and should be retried rather than counted as wrong answers.
            _stripped = response_content.strip()
            if _stripped.startswith("{"):
                try:
                    _maybe_err = json.loads(_stripped)
                    if isinstance(_maybe_err, dict):
                        if _maybe_err.get("object") == "error":
                            raise Exception(f"API returned error object in content: {_stripped[:200]}")
                        if isinstance(_maybe_err.get("error"), dict):
                            raise Exception(f"API returned error dict in content: {_stripped[:200]}")
                except json.JSONDecodeError:
                    pass  # Not JSON — fine, continue normally

            # Detect plaintext server-error strings that vLLM can inject
            _lower = _stripped[:120].lower()
            _error_phrases = [
                "internal server error",
                "bad gateway",
                "service unavailable",
                "context length exceeded",
            ]
            if any(phrase in _lower for phrase in _error_phrases):
                raise Exception(f"API returned error text in content: {_stripped[:200]}")

            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": response.usage.prompt_tokens,
                "response_num_tokens": response.usage.completion_tokens,
            }
            
            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            return response_content, call_info
            
        except Exception as e:
            # Check for both timeout and rate limit errors
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()
            
            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, 'response'):
                try:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(f"[{role.upper()}] Server error detected: HTTP {status_code}")
                except:
                    pass
            
            # Truncated output (finish_reason=length) is permanent — the prompt is too
            # long or max_tokens too small.  Retrying will never succeed, so return
            # immediately as an incorrect response rather than entering the retry loop.
            is_length_truncation = "response truncated (finish_reason=length)" in str(e).lower()
            if is_length_truncation:
                print(f"[{role.upper()}] Output truncated (finish_reason=length) for {call_id} — skipping sample")
                error_time = time.time()
                call_info = {
                    "role": role,
                    "call_id": call_id,
                    "model": model,
                    "prompt": prompt,
                    "error": "LENGTH_TRUNCATION: " + str(e),
                    "total_time": error_time - start_time,
                    "prompt_length": len(prompt),
                    "response_length": 0,
                    "length_truncation": True,
                }
                if log_dir:
                    log_llm_call(log_dir, call_info)
                incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                return incorrect_response, call_info

            # Also check for 500 errors in the error message itself, plus our own
            # content-level error/truncation exceptions raised above.
            if any(k in str(e).lower() for k in [
                "500 internal server error", "internal server error",
                "502 bad gateway", "503 service unavailable",
                "api returned error object in content",
                "api returned error dict in content",
                "api returned error text in content",
                "context length exceeded",
            ]):
                is_server_error = True
                print(f"[{role.upper()}] Server error detected in message: {str(e)[:100]}...")
            
            # Also check for specific OpenAI exceptions
            if hasattr(openai, 'RateLimitError') and isinstance(e, openai.RateLimitError):
                is_rate_limit = True
            
            # Check for OpenAI InternalServerError
            if hasattr(openai, 'InternalServerError') and isinstance(e, openai.InternalServerError):
                is_server_error = True
                print(f"[{role.upper()}] OpenAI InternalServerError detected")
            
            # Empty response: retry up to 3 times before giving up.
            # We must not build graphs or score accuracy on empty content.
            # vLLM returning None content is almost always transient.
            if is_empty_response:
                if attempt <= 3:
                    sleep_time = min(sleep_seconds, 10) * random.uniform(0.5, 1.5)
                    print(f"[{role.upper()}] Empty response for {call_id} "
                          f"(attempt {attempt}/3), retrying in {sleep_time:.1f}s...")
                    log_problematic_request(call_id, prompt, model, api_params, e, log_dir,
                                            using_key_mixer, client if using_key_mixer else None)
                    attempt += 1
                    time.sleep(sleep_time)
                    continue
                # All 3 attempts exhausted — raise so callers can exclude this
                # sample from accuracy math (test eval skips it; training skips the step).
                print(f"[{role.upper()}] Empty response for {call_id} after 3 attempts — giving up")
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir,
                                        using_key_mixer, client if using_key_mixer else None)
                raise e
            
            # Retry logic for timeouts, rate limits, and server errors
            if (is_timeout or is_rate_limit or is_server_error) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                elif is_empty_response:
                    error_type = "returned empty response"
                    base_sleep = sleep_seconds
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                      f"({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            raise e
