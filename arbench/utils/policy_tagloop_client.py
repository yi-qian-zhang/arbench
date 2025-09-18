# arbench/utils/policy_tagloop_client.py
import os
import json
import time
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL", "http://localhost:8722/v1")
POLICY_API_KEY  = os.getenv("POLICY_API_KEY", "")

class TagLoopPolicyClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        stop: Optional[list] = None,
        max_tokens: int = 4096,
        timeout: int = 60,
        include_stop_str_in_output: bool = True,
        
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        max_retries: int = 3,
        retry_wait: int = 1,
    ):
        self.model = model
        self.base_url = (base_url or POLICY_BASE_URL).rstrip("/")
        self.api_key = api_key or POLICY_API_KEY
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop or ["</asking>", "<｜end▁of▁sentence｜>"]
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.include_stop_str_in_output = include_stop_str_in_output
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: List[Dict]) -> Tuple[str, Optional[str]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "add_generation_prompt": self.add_generation_prompt,
            "continue_final_message": self.continue_final_message,
            "echo": False,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    resp = r.json()
                    content = resp["choices"][0]["message"]["content"]
                    stop_reason = resp["choices"][0].get("stop_reason")
                    return content, stop_reason
                else:
                    last_err = f"HTTP {r.status_code}: {r.text}"
            except Exception as e:
                last_err = str(e)

            if attempt < self.max_retries:
                time.sleep(self.retry_wait)

        raise RuntimeError(f"policy chat failed: {last_err}")
