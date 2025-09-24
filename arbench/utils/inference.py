# """
# Inference utilities for AR-Bench

# This module provides inference capabilities for different LLM providers including OpenAI and Together AI.
# """
# import os
# from typing import Dict, List, Optional

# import numpy as np
# from openai import OpenAI


# _default_model = "gpt-4o"
# _default_temperature = 0.7
# _default_top_p = 0.7

# def inference(messages: List[Dict], model: Optional[str] = None, json_format: bool = False, api_key: Optional[str] = None, base_url: Optional[str] = None,
#               temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:

#     try:
#         # Use global defaults if parameters are not specified
#         if model is None:
#             model = _default_model
#         if temperature is None:
#             temperature = _default_temperature
#         if top_p is None:
#             top_p = _default_top_p
            
#         client = OpenAI(
#             base_url=base_url,
#             api_key=api_key
#         )
        
#         kwargs = {
#             "model": model,
#             "messages": messages,
#             "temperature": temperature,
#             "top_p": top_p
#         }
        
#         if json_format:
#             kwargs["response_format"] = {"type": "json_object"}
        
#         response = client.chat.completions.create(**kwargs)
#         return response or "{}"
#     except Exception as e:
#         print(f"Inference error: {e}")
#         return "{}"



"""
Inference utilities for AR-Bench

This module provides inference capabilities for different LLM providers including OpenAI and Together AI.
"""
import os
from typing import Dict, List, Optional
from openai import OpenAI
from types import SimpleNamespace

_default_model = "gpt-4o"
_default_temperature = 0.7
_default_top_p = 0.7

def _fake_response(content: str) -> object:
    """
    构造一个假的 ChatCompletion-like 对象，保证有 .choices[0].message.content
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    )

def inference(
    messages: List[Dict],
    model: Optional[str] = None,
    json_format: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
):
    try:
        # Use global defaults if parameters are not specified
        if model is None:
            model = _default_model
        if temperature is None:
            temperature = _default_temperature
        if top_p is None:
            top_p = _default_top_p

        client = OpenAI(base_url=base_url, api_key=api_key)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }

        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"[Inference error] {e}")
        # 返回一个假的响应，避免 AttributeError
        return _fake_response("<<ERROR: inference failed or was filtered>>")
