# arbench/utils/interactive_policy.py
import re
import time
import json
from typing import Callable, Dict, List, Optional, Tuple
import requests

STOP_TOKENS_DEFAULT = ["</asking>", "<｜end▁of▁sentence｜>"]

def _normalize_base_url(base_url: str) -> str:
    """
    兼容以下输入：
      - http://host:port
      - http://host:port/v1
      - http://host:port/v1/
      - http://host:port/v1/chat/completions
    最终统一成：.../v1/chat/completions
    """
    u = (base_url or "").rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    if u.endswith("/v1/"):
        return u + "chat/completions"
    # 既没有 /v1 也没有 /chat/completions，就补成 /v1/chat/completions
    return u + "/v1/chat/completions"

def _extract_between(s: str, left: str, right: str) -> Optional[str]:
    m = re.search(re.escape(left) + r"(.*?)" + re.escape(right), s, re.S)
    return m.group(1).strip() if m else None

def _extract_visible_after_think(s: str) -> str:
    """
    从 <think>…</think> 后截取“可见回复”。
    若没有 </think>，退而求其次，返回整段去首尾空白。
    """
    if "</think>" in s:
        return s.split("</think>", 1)[1].strip()
    return s.strip()

def infer_target_name(question: str, candidates: List[str], default_name: str = "") -> str:
    """
    解析 <asking> 中的问题要问谁。
    策略：
      1. 问题文本里若直接包含某个候选名字（大小写不敏感），匹配之。
      2. 否则回落到 default_name（上一轮选中的嫌疑人）。
      3. 实在不行返回 "UNKNOWN"。
    """
    qlow = question.lower()
    for n in candidates:
        if n.lower() in qlow:
            return n
    if default_name:
        return default_name
    return "UNKNOWN"

class RespondRecorder:
    """可选：记录 <think> 期间的中间问答，便于导出 """
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
    让“策略模型”走 <think>/<asking>/<response> 循环。
    其核心是：当生成遇到 stop token '</asking>' 时，取出 <asking>...，
    通过 ask_router(question) 调用“用户模拟器/回复模型”得到 <response>，
    把 <response> 填回 <think> 链，然后继续生成，直到生成出 </think> 后的最终可见答案。
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
                # 如果后端是 vLLM，会返回 200 + OpenAI 兼容格式；404/5xx 则抛异常
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)
        # 最后一次失败
        if hasattr(last_err, "response") and getattr(last_err.response, "text", None):
            raise RuntimeError(f"PolicyThinkRunner HTTP error: {last_err} | {last_err.response.text}")
        raise RuntimeError(f"PolicyThinkRunner HTTP error: {last_err}")

    def _chat_once(self, messages: List[Dict], add_assistant_prefix: str) -> Tuple[str, int, int, Optional[str]]:
        """
        发起一次补全。将 <think> 累积内容作为最后一条 assistant 消息传入，以便“续写”。
        这里依赖 vLLM 的 `continue_final_message=True` 和 OpenAI 兼容接口。
        返回：新增文本增量、输入token、输出token、stop_reason(如有)
        """
        payload = {
            "model": self.model,
            "messages": messages + [{"role": "assistant", "content": add_assistant_prefix}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            # 下面这几个是 vLLM 扩展参数，OpenAI 官方会忽略，但 vLLM 会吃：
            "continue_final_message": True,
            "include_stop_str_in_output": True,
            "add_generation_prompt": False,
            "echo": False,
        }
        data = self._post(payload)
        choice = data["choices"][0]
        delta = choice["message"]["content"]
        # vLLM 里字段可能叫 stop_reason（也可能是 finish_reason）
        stop_reason = choice.get("stop_reason", choice.get("finish_reason"))
        in_tok = data.get("usage", {}).get("prompt_tokens", 0)
        out_tok = data.get("usage", {}).get("completion_tokens", 0)
        return delta, in_tok, out_tok, stop_reason

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
        history_until_assistant: 例如：[{"role":"system",...},{"role":"user",...}]，
            也就是到“需要 assistant 说话”之前的上下文。
        suspect_names: 可用来路由 <asking> 的目标人物。
        ask_router:   真正把 <asking> 发给“回复模型(gpt-4o)”的回调。
        返回：最终可见答案（</think> 之后的内容），累计输入token，累计输出token。
        """
        if recorder is None:
            recorder = RespondRecorder()

        total_in, total_out = 0, 0
        acc = "<think>\n"   # 累积 assistant 的“思考链”
        while True:
            delta, in_tok, out_tok, stop_reason = self._chat_once(history_until_assistant, acc)
            total_in += in_tok
            total_out += out_tok
            acc += delta

            # 如果这一步“停在了 </asking>”，就触发“问用户模拟器”
            # 1) vLLM 若设置 include_stop_str_in_output=True，content 里会包含 '</asking>'
            # 2) 若没包含，也可以用 stop_reason 比较保险
            hit_asking = ("</asking>" in acc) or (stop_reason == "</asking>")
            if hit_asking:
                q = _extract_between(acc, "<asking>", "</asking>")
                if q:
                    # 路由到正确嫌疑人
                    target = infer_target_name(q, suspect_names, default_target_name)
                    if target == "UNKNOWN":
                        # 给个兜底回答，避免死循环
                        target = default_target_name or (suspect_names[0] if suspect_names else "UNKNOWN")
                    # 调用户模拟器（= 回复模型，比如 gpt-4o）
                    routed_target, a = ask_router(q)

                    # 记录（可选）
                    recorder.record("assistant(thinking.asking)", q, {"target": routed_target})
                    recorder.record("user_simulator(response)", a, {"target": routed_target})

                    # 把答复补回 <think> 链
                    acc += f"\n<response>{a}</response>"
                    # 继续 while 循环（再次调用策略模型续写）
                    continue

            # 如果已经写完 </think>，就可以抽取最终可见答案返回了
            if "</think>" in acc:
                visible = _extract_visible_after_think(acc)
                return visible, total_in, total_out

            # 若遇到 “全停”(例如遇到 <｜end▁of▁sentence｜>) 却还没 </think>，
            # 通常是提示不规范或模型异常，尽量返回当前可见片段，避免死循环。
            if stop_reason and stop_reason != "</asking>":
                visible = _extract_visible_after_think(acc)
                return visible, total_in, total_out
