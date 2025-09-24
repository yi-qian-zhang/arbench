# arbench/utils/interactive_policy.py
# -*- coding: utf-8 -*-

import re
import time
import json
from typing import Callable, Dict, List, Optional, Tuple
import requests

# 模型思维链中的停靠标记：
# - "</asking>"：触发向“用户模拟器/回复模型”提问
# - "<｜end▁of▁sentence｜>"：很多 Qwen/R1 系列在 SFT 时会用到的“句终”token 对应的可见字符串
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
    return u + "/v1/chat/completions"

def _extract_between(s: str, left: str, right: str) -> Optional[str]:
    m = re.search(re.escape(left) + r"(.*?)" + re.escape(right), s, re.S)
    return m.group(1).strip() if m else None

def _extract_visible_after_think(s: str) -> str:
    """
    从 </think> 之后截取“可见回复”。
    若没有 </think>，退而求其次，返回整段去首尾空白。
    """
    if "</think>" in s:
        return s.split("</think>", 1)[1].strip()
    return s.strip()

def infer_target_name(question: str, candidates: List[str], default_name: str = "") -> str:
    """
    解析 <asking> 中的问题要问谁。
    策略：
      1) 问题文本里若直接包含某个候选名字（大小写不敏感），匹配之。
      2) 否则回落到 default_name（上一轮选中的嫌疑人）。
      3) 实在不行返回 "UNKNOWN"。
    """
    qlow = question.lower()
    for n in candidates:
        if n.lower() in qlow:
            return n
    if default_name:
        return default_name
    return "UNKNOWN"

class RespondRecorder:
    """可选：记录 <think> 期间的中间问答，便于导出/排障。"""
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
    只用“手工续写”的策略推理器（不依赖 vLLM 的 continue_final_message）。
    逻辑：
      - 累积一个 assistant 的 <think> 串 acc（初始为 "<think>\\n"）
      - 每一步把 acc 作为“最后一条 assistant 消息”传给模型，请它继续写“下一条回复”
      - 拿到新增片段 delta，把它拼到 acc 里
      - 若 acc 中出现 </asking>，抽出 <asking>...，通过 ask_router() 去问“用户模拟器”，
        再把 <response>...</response> 插回 acc，继续循环
      - 直到 acc 出现 </think>，或遇到其他停靠/用完 token 为止
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
        # 手工续写相关配置
        total_max_tokens: int = 4096,    # 本轮手工续写的总生成预算（近似“从 <think> 到结束”的预算）
        step_max_tokens: int = 1024,     # 每次调用接口的最大生成步长
        include_stop_str_in_output: bool = True,  # 让 vLLM 在输出里包含触发的 stop 字符串（便于“看到”</asking>）
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

        self.total_max_tokens = max(1, int(total_max_tokens))
        self.step_max_tokens = max(1, int(step_max_tokens))
        self.include_stop_str_in_output = include_stop_str_in_output

    # ---------- HTTP ----------
    def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)

        # 最后一次失败：补充可读提示
        if hasattr(last_err, "response") and getattr(last_err.response, "text", None):
            msg = (
                f"PolicyThinkRunner HTTP error: {last_err} | {last_err.response.text}\n\n"
                "[Fix Hint] 你的策略模型 chat_template 若与“原生续写”相关参数冲突，"
                "请改用“手工续写”（本文件已经禁用 continue_final_message），或检查服务端是否支持传入的扩展参数。"
            )
            raise RuntimeError(msg)
        raise RuntimeError(f"PolicyThinkRunner HTTP error: {last_err}")

    def _chat_once(
        self,
        messages_until_assistant: List[Dict],
        accumulated_assistant_content: str,
        tokens_budget_left: int,
    ) -> Tuple[str, int, int, Optional[str]]:
        """
        发起一次“手工续写”的补全：
        - 不使用 continue_final_message
        - 把当前累计的 <think> 文本作为“上一条 assistant 消息”传入
        - 让模型生成“下一条 assistant”（新的片段），我们把它当作 delta 拼回去
        返回： (delta, prompt_tokens, completion_tokens, stop_reason)
        """
        # 每步的生成步长不超过 step_max_tokens，也不超过剩余额度
        step_tokens = max(1, min(self.step_max_tokens, tokens_budget_left))

        payload = {
            "model": self.model,
            "messages": messages_until_assistant + [
                {"role": "assistant", "content": accumulated_assistant_content}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "max_tokens": step_tokens,
            # OpenAI 兼容字段（vLLM 扩展的会被忽略或生效，但这里**不使用** continue_final_message）
            "add_generation_prompt": False,
            "echo": False,
        }

        # vLLM 的扩展：包含触发的 stop 字符串在输出里，便于我们看到 </asking>
        if self.include_stop_str_in_output:
            payload["include_stop_str_in_output"] = True

        data = self._post(payload)
        choice = data["choices"][0]
        delta = choice["message"]["content"] or ""
        stop_reason = choice.get("stop_reason", choice.get("finish_reason"))
        in_tok = data.get("usage", {}).get("prompt_tokens", 0)
        out_tok = data.get("usage", {}).get("completion_tokens", 0)
        return delta, in_tok, out_tok, stop_reason

    def run(
        self,
        history_until_assistant: List[Dict],
        suspect_names: List[str],
        ask_router: Callable[[str], Tuple[str, str]],  # q -> (target_name, answer)
        recorder: Optional[RespondRecorder] = None,
        default_target_name: str = "",
    ) -> Tuple[str, int, int]:
        """
        手工续写版：只处理“新出现”的 <asking>…</asking>，并维护总 token 预算。
        """
        if recorder is None:
            recorder = RespondRecorder()

        total_in, total_out = 0, 0
        acc = "<think>\n"
        consumed_pos = 0                  # 已处理到的位置
        safety_guard = 0                  # 极限保护
        tokens_left = self.total_max_tokens  # 总生成预算

        while tokens_left > 0:
            # 1) 按剩余额度发起一步生成（不使用 continue_final_message）
            delta, in_tok, out_tok, stop_reason = self._chat_once(
                history_until_assistant, acc, tokens_left
            )
            total_in += in_tok
            total_out += out_tok
            tokens_left = max(0, tokens_left - (out_tok or 0))

            acc += (delta or "")

            # 2) 在“未消费”的区域寻找新闭合的 </asking>，只处理最近这一对
            idx_close = acc.find("</asking>", consumed_pos)
            if idx_close != -1:
                idx_open = acc.rfind("<asking>", consumed_pos, idx_close)
                if idx_open != -1:
                    q = acc[idx_open + len("<asking>"): idx_close].strip()

                    # 空问句：直接删掉这对标签，避免死循环
                    if not q:
                        acc = acc[:idx_open] + acc[idx_close + len("</asking>"):]
                        continue

                    # 路由 & 询问用户模拟器
                    target = infer_target_name(q, suspect_names, default_target_name)
                    if target == "UNKNOWN":
                        target = default_target_name or (suspect_names[0] if suspect_names else "UNKNOWN")
                    routed_target, a = ask_router(q)

                    recorder.record("assistant(thinking.asking)", q, {"target": routed_target})
                    recorder.record("user_simulator(response)", a, {"target": routed_target})

                    # 在 </asking> 后插入 <response>…</response>，并推进游标
                    injection = f"\n<response>{a}</response>"
                    insert_at = idx_close + len("</asking>")
                    acc = acc[:insert_at] + injection + acc[insert_at:]
                    consumed_pos = insert_at + len(injection)

                    safety_guard += 1
                    if safety_guard > 256:
                        visible = _extract_visible_after_think(acc)
                        return visible, total_in, total_out

                    # 处理完这一问，立刻进入下一轮，让模型接着续写
                    continue

            # 3) 若已闭合 </think>，抽出可见回复
            if "</think>" in acc:
                visible = _extract_visible_after_think(acc)
                return visible, total_in, total_out

            # 4) 命中其他停靠（如 <｜end▁of▁sentence｜> / stop / eos），也返回当前可见片段
            if stop_reason and stop_reason != "</asking>":
                visible = _extract_visible_after_think(acc)
                return visible, total_in, total_out

        # 5) token 预算用尽，返回当前可见片段
        visible = _extract_visible_after_think(acc)
        return visible, total_in, total_out
