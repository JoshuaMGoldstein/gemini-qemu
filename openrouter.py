import base64
import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _headers() -> Dict[str, str]:
    api_key = os.environ.get("LLMVM_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("LLMVM_OPENROUTER_API_KEY or OPENROUTER_API_KEY not set. Put it in .env or env.")
    site_url = os.environ.get("OPENROUTER_SITE_URL", "http://localhost")
    app_name = os.environ.get("OPENROUTER_APP_NAME", "llm-vm-controller")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": site_url,
        "X-Title": app_name,
    }


def _image_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode('ascii')
    # Detect image format from the first few bytes
    if img_bytes.startswith(b'\xff\xd8\xff'):
        mime_type = "image/jpeg"
    elif img_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        mime_type = "image/png"
    else:
        mime_type = "image/png"  # default fallback
    return f"data:{mime_type};base64,{b64}"


def call_chat_vision(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    image_bytes: bytes,
    temperature: float = 0.2,
    extra: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Call OpenRouter Chat Completions with a text+image message.

    Returns the parsed JSON response from OpenRouter.
    """
    image_url = _image_to_data_url(image_bytes)
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }
    # Add Fireworks provider specifically for Qwen 2.5 VL 32B model
    if model == "qwen/qwen2.5-vl-32b-instruct":
        payload["provider"] = {
            "order": ["fireworks"],
            "allow_fallbacks": False
        }
    if extra:
        payload.update(extra)
    # Try provided base URL, then fall back to alternate paths if 404
    tried: list[str] = []
    base = (base_url or DEFAULT_BASE_URL).rstrip('/')
    
    # Build list of URLs to try
    urls_to_try = []
    
    # Always use the correct endpoint first
    fallback = "https://openrouter.ai/api/v1/chat/completions"
    
    # If caller provided a full path ending in /chat/completions, use it as-is
    if base.endswith('/chat/completions'):
        urls_to_try.append(base)
    # If it ends with /v1, append /chat/completions
    elif base.endswith('/v1'):
        urls_to_try.append(f"{base}/chat/completions")
    # If it ends with /api, append /v1/chat/completions
    elif base.endswith('/api'):
        urls_to_try.append(f"{base}/v1/chat/completions")
    # Otherwise append /api/v1/chat/completions
    else:
        urls_to_try.append(f"{base}/api/v1/chat/completions")
    
    # Add fallback URL if not already in list
    if fallback not in urls_to_try:
        urls_to_try.append(fallback)
    
    for url in urls_to_try:
        if url in tried:
            continue
        tried.append(url)
        headers = _headers()
        print(f"🌐 Trying OpenRouter URL: {url}")
        print(f"🔑 Headers: {dict(headers)}")
        print(f"📦 Payload model: {payload.get('model', 'N/A')}")
        print(f"📦 Payload keys: {list(payload.keys())}")
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"🔍 Response status: {resp.status_code}")
        print(f"🔍 Response headers: {dict(resp.headers)}")
        print(f"🔍 Response body (first 500 chars): {resp.text[:500]}")
        if resp.status_code == 404:
            # Try next candidate
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Include body snippet for easier debugging
            snippet = resp.text[:500]
            raise requests.HTTPError(f"OpenRouter error {resp.status_code}: {snippet}", response=resp) from e
        # Ensure JSON response
        ctype = resp.headers.get('Content-Type', '')
        if 'application/json' not in ctype:
            snippet = resp.text[:500]
            raise requests.HTTPError(
                f"OpenRouter returned non-JSON (Content-Type: {ctype}). URL={url}. Body snippet: {snippet}",
                response=resp,
            )
        return resp.json()
    # If we get here, all candidates returned 404
    raise requests.HTTPError(f"All OpenRouter endpoints returned 404. Tried: {tried}")


def call_chat(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.2,
    extra: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Call OpenRouter Chat Completions with text-only message (no vision).

    Returns the parsed JSON response from OpenRouter.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    # Add Fireworks provider specifically for Qwen 2.5 VL 32B model
    if model == "qwen/qwen2.5-vl-32b-instruct":
        payload["provider"] = {
            "order": ["fireworks"],
            "allow_fallbacks": False
        }
    if extra:
        payload.update(extra)
    
    # Try provided base URL, then fall back to alternate paths if 404
    tried: list[str] = []
    base = (base_url or DEFAULT_BASE_URL).rstrip('/')
    
    # Build list of URLs to try
    urls_to_try = []
    
    # Always use the correct endpoint first
    fallback = "https://openrouter.ai/api/v1/chat/completions"
    
    # If caller provided a full path ending in /chat/completions, use it as-is
    if base.endswith('/chat/completions'):
        urls_to_try.append(base)
    # If it ends with /v1, append /chat/completions
    elif base.endswith('/v1'):
        urls_to_try.append(f"{base}/chat/completions")
    # If it ends with /api, append /v1/chat/completions
    elif base.endswith('/api'):
        urls_to_try.append(f"{base}/v1/chat/completions")
    # Otherwise append /api/v1/chat/completions
    else:
        urls_to_try.append(f"{base}/api/v1/chat/completions")
    
    # Add fallback URL if not already in list
    if fallback not in urls_to_try:
        urls_to_try.append(fallback)
    
    for url in urls_to_try:
        if url in tried:
            continue
        tried.append(url)
        headers = _headers()
        print(f"🌐 Trying OpenRouter URL: {url}")
        print(f"🔑 Headers: {dict(headers)}")
        print(f"📦 Payload model: {payload.get('model', 'N/A')}")
        print(f"📦 Payload keys: {list(payload.keys())}")
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"🔍 Response status: {resp.status_code}")
        print(f"🔍 Response headers: {dict(resp.headers)}")
        print(f"🔍 Response body (first 500 chars): {resp.text[:500]}")
        if resp.status_code == 404:
            # Try next candidate
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Include body snippet for easier debugging
            snippet = resp.text[:500]
            raise requests.HTTPError(f"OpenRouter error {resp.status_code}: {snippet}", response=resp) from e
        # Ensure JSON response
        ctype = resp.headers.get('Content-Type', '')
        if 'application/json' not in ctype:
            snippet = resp.text[:500]
            raise requests.HTTPError(
                f"OpenRouter returned non-JSON (Content-Type: {ctype}). URL={url}. Body snippet: {snippet}",
                response=resp,
            )
        return resp.json()
    # If we get here, all candidates returned 404
    raise requests.HTTPError(f"All OpenRouter endpoints returned 404. Tried: {tried}")
