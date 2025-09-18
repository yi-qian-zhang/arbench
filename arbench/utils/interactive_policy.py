# arbench/utils/interactive_policy.py
import re
import time
import json
from typing import Callable, Dict, List, Optional, Tuple
import requests

# 注意：不包含 </think>，否则会被提前截断
STOP_TOKENS_DEFAULT = ["</asking>", "<｜end▁of▁sentence｜>"]

def _normalize_base_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    if u.endswith("/v1/"):
        return u + "chat/completions"
    return u + "/v1/chat/completions"

def _extract_between(s: str, left: str, right: str) -> Optional[str]:
    m = re.search(re.escape(left) + r"(.*?)" + re.escape(right), s, re.S)
    return m.group(1).strip() if m else None

def _extract_visible_after_think(s: str) -> str:
    # 最终返回“</think>之后的可见内容”；若没有 </think>，就返回整体（避免极端 prompt 下也能产出）
    if "</think>" in s:
        return s.split("</think>", 1)[1].strip()
    return s.strip()

def infer_target_name(question: str, candidates: List[str], default_name: str = "") -> str:
    qlow = question.lower()
    for n in candidates:
        if n.lower() in qlow:
            return n
    if default_name:
        return default_name
    return "UNKNOWN"

class RespondRecorder:
    def __init__(self, system_prompts: Optional[Dict[str, str]] = None):
        self.logs = []
        self.system_prompts = system_prompts or {}

    def record(self, role: str, content: str, meta: Optional[Dict] = None):
        self.logs.append({
            "role": role,
            "content": content,
            "meta": meta or {}
        })

class PolicyThinkRunner:
    """
    原生续写版：
    - 用 continue_final_message=True 从最后一条 assistant 内容处“接着写”；
    - stop 只包含 </asking> 和模型的 EOS（不要放 </think>）；
    - 命中 </asking> 就 ask_router 取答复，回填 <response>…</response>，继续写；
    - 直到：模型自然停（EOS/finish_reason）或 max_completion_tokens 用完为止；
    - 最终把 </think> 之后的“可见文本”返回（即使中途没遇到 </think>，也会返回当前累计）。
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.7,
        stop_tokens: Optional[List[str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_wait: int = 3,
        max_completion_tokens: int = 4096,   # 单轮总预算
        eos_aliases: Optional[List[str]] = None,  # 某些 vLLM 版本可能返回 'stop'/'length'
    ):
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or "none"
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.stop_tokens = stop_tokens or STOP_TOKENS_DEFAULT
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.max_completion_tokens = max_completion_tokens
        self.eos_aliases = set((eos_aliases or []) + ["stop", "length", None])

        # 运行期变量
        self._remaining_tokens = max_completion_tokens

    # ---------- HTTP ----------
    def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)

        detail = ""
        if hasattr(last_err, "response") and getattr(last_err.response, "text", None):
            detail = last_err.response.text
        msg = f"PolicyThinkRunner HTTP error: {last_err}"
        if detail:
            msg += f" | {detail}"
        if "continue_final_message is set" in detail:
            msg += (
                "\n\n[Fix Hint] 你的策略模型 chat_template 没有保留最后一条 assistant 内容。\n"
                "为使用原生续写，请确保渲染后的最后一条消息是 role='assistant' 且内容不被模板吞掉。"
            )
        raise RuntimeError(msg)

    def _chat_once(
        self,
        messages_before_assistant: List[Dict],
        assistant_prefix: str
    ) -> Tuple[str, int, int, Optional[str]]:
        # 每次调用根据剩余额度设置 max_completion_tokens
        # 有些 vLLM 版本用 max_tokens，为兼容你也可以加一行 payload["max_tokens"]=...
        payload = {
            "model": self.model,
            "messages": messages_before_assistant + [
                {"role": "assistant", "content": assistant_prefix}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "continue_final_message": True,
            "include_stop_str_in_output": True,
            "add_generation_prompt": False,
            "echo": False,
            "max_completion_tokens": max(1, self._remaining_tokens),
        }
        data = self._post(payload)
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "") or ""
        stop_reason = choice.get("stop_reason", choice.get("finish_reason"))

        usage = data.get("usage", {}) or {}
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        # 扣减剩余额度
        self._remaining_tokens = max(0, self._remaining_tokens - int(out_tok or 0))

        return content, in_tok, out_tok, stop_reason

    # ---------- Run Loop ----------
    def run(
        self,
        history_until_assistant: List[Dict],
        suspect_names: List[str],
        ask_router: Callable[[str], Tuple[str, str]],  # q -> (target_name, answer)
        recorder: Optional[RespondRecorder] = None,
        default_target_name: str = "",
    ) -> Tuple[str, int, int]:
        """
        返回：(最终可见答案, 累计输入 token, 累计输出 token)
        """
        if recorder is None:
            recorder = RespondRecorder()

        # 每次 run 重置预算（防止多轮调用共享旧值）
        self._remaining_tokens = self.max_completion_tokens

        total_in, total_out = 0, 0
        completion = "<think>\n"  # 模型“思考草稿”累积（最后一条 assistant）

        while True:
            if self._remaining_tokens <= 0:
                # 预算耗尽，直接返回当前可见
                return _extract_visible_after_think(completion), total_in, total_out

            delta, in_tok, out_tok, stop_reason = self._chat_once(history_until_assistant, completion)
            total_in += in_tok
            total_out += out_tok
            completion += delta

            # 命中 </asking> ：取问题 -> 调“用户模拟器” -> 回填 <response> -> 继续原生续写
            hit_asking = ("</asking>" in completion) or (stop_reason == "</asking>")
            if hit_asking:
                q = _extract_between(completion, "<asking>", "</asking>")
                if q:
                    target = infer_target_name(q, suspect_names, default_target_name)
                    if target == "UNKNOWN":
                        target = default_target_name or (suspect_names[0] if suspect_names else "UNKNOWN")
                    routed_target, a = ask_router(q)
                    recorder.record("assistant(thinking.asking)", q, {"target": routed_target})
                    recorder.record("user_simulator(response)", a, {"target": routed_target})
                    completion += f"\n<response>{a}</response>"
                    # 不返回，继续 while 循环续写
                    continue

            # 若不是 </asking> 打断，检查是否“自然停”或“长度停”
            # vLLM 常见：stop_reason in {"stop","length",None}；或者匹配到 EOS（我们已把 EOS 放在 stop 里）
            if stop_reason in self.eos_aliases or self._remaining_tokens <= 0:
                return _extract_visible_after_think(completion), total_in, total_out

            # 保险：如果遇到未知 stop_reason，但不是 </asking>，也认为已停
            if stop_reason and stop_reason != "</asking>":
                return _extract_visible_after_think(completion), total_in, total_out
