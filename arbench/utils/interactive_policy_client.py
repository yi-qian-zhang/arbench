# interactive_policy_client.py
import re
from typing import Dict, List, Optional, Tuple
import requests
import json

class InteractivePolicyClient:
    """处理distill模型的交互式问答"""
    
    def __init__(
        self, 
        model_path: str,
        base_url: str,
        api_key: str = "none",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        timeout: int = 60
    ):
        self.model_path = model_path
        self.base_url = base_url.rstrip('/') + '/chat/completions'
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.completion = ""
        self.stop_reason = None
        
    def generate_with_interaction(
        self, 
        system_prompt: str,
        user_prompt: str,
        response_model: str,
        response_api_key: str,
        response_base_url: str,
        response_temperature: float = 0.7,
        response_top_p: float = 0.7,
        case_context: Dict = None
    ) -> Tuple[str, str, List[Dict], int, int]:

        # 初始化completion，以<think>开始
        self.completion = "<think>\n"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": self.completion}
        ]
        
        interaction_history = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 调用distill模型
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "stop": ["</asking>", "<｜end▁of▁sentence｜>"],  生成完
            "stop": ["</asking>", "</think>"],
            "max_completion_tokens": self.max_tokens,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "include_stop_str_in_output": True,
            "echo": False
        }
        
        try:
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            response_json = response.json()
            
            model_response = response_json['choices'][0]['message']['content']
            self.completion += model_response
            self.stop_reason = response_json['choices'][0].get('stop_reason', 
                                response_json['choices'][0].get('finish_reason'))
            
            total_input_tokens += response_json['usage']['prompt_tokens']
            total_output_tokens += response_json['usage']['completion_tokens']
            
            # 检查stop token
            if "</asking>" in model_response:
                self.stop_reason = "</asking>"
            
        except Exception as e:
            print(f"Error calling distill model: {e}")
            return user_prompt, self.completion, [], 0, 0
        
        # 交互循环
        while self.stop_reason == "</asking>":
            # 提取助手的询问
            ask_match = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL)
            if not ask_match:
                break
                
            ask_content = ask_match.group(1).strip()
            
            # 使用GPT-4O生成回复
            gpt_response = self._get_gpt_response(
                ask_content, 
                case_context,
                response_model,
                response_api_key,
                response_base_url,
                response_temperature,
                response_top_p
            )
            
            # 记录交互
            interaction_history.append({
                "asking": ask_content,
                "response": gpt_response
            })
            
            # 把response插入think继续生成
            self.completion += f"\n<response>{gpt_response}</response>\n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.completion}
            ]
            
            payload["messages"] = messages
            payload["max_completion_tokens"] = self.max_tokens - total_output_tokens
            
            try:
                response = requests.post(
                    url=self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response_json = response.json()
                
                model_response = response_json['choices'][0]['message']['content']
                self.completion += model_response
                self.stop_reason = response_json['choices'][0].get('stop_reason',
                                    response_json['choices'][0].get('finish_reason'))
                
                if "</asking>" in model_response:
                    self.stop_reason = "</asking>"
                
                total_input_tokens += response_json['usage']['prompt_tokens']
                total_output_tokens += response_json['usage']['completion_tokens']
                
            except Exception as e:
                print(f"Error in interaction loop: {e}")
                break
        
        # 确保completion以</think>结尾并提取最终答案
        if "</think>" not in self.completion:
            self.completion += "\n</think>\n"
        
        final_answer = self._extract_final_answer(self.completion)
        
        return final_answer, self.completion, interaction_history, total_input_tokens, total_output_tokens
    
    def _get_gpt_response(
        self, 
        question: str, 
        context: Dict,
        response_model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        top_p: float
    ) -> str:
        """使用GPT-4O生成对侦探询问的回复"""
        from arbench.utils.inference import inference
        
        # 构建GPT的系统提示
        gpt_system = """你是一个协助侦探破案的助手。侦探会询问你关于案件调查策略的建议。
        请根据案件背景提供简洁、有用的建议，帮助侦探更好地进行调查。
        你的回答应该：
        1. 简洁明了，1-2句话
        2. 基于案件事实和逻辑推理
        3. 提供具体可行的建议"""
        
        # 构建上下文信息
        case_info = ""
        if context:
            if 'case_info' in context:
                case_info = f"案件背景：{context['case_info']}\n"
            if 'suspect' in context:
                case_info += f"当前讨论的嫌疑人：{context['suspect']}\n"
            if 'turn' in context:
                case_info += f"当前是第{context['turn']}轮询问\n"
            if 'record' in context and context['record']:
                case_info += f"已有调查记录：{context['record'][:200]}...\n"
        
        gpt_messages = [
            {"role": "system", "content": gpt_system},
            {"role": "user", "content": f"{case_info}\n侦探的问题：{question}"}
        ]
        
        try:
            # 调用GPT-4O
            response = inference(
                gpt_messages, 
                model=response_model,
                temperature=temperature,
                top_p=top_p,
                api_key=api_key,
                base_url=base_url
            )
            
            # --- 新增的健壮性检查 ---
            if (response and response.choices and 
                response.choices[0].message and 
                response.choices[0].message.content):
                
                return response.choices[0].message.content.strip()
            else:
                # API调用失败, 返回空, 或被内容过滤
                print(f"警告: 响应模型 (GPT-4O) 返回了无效内容。")
                print(f"API 响应详情: {response}")
                return "我目前无法回答这个问题。" # 返回一个安全的默认字符串

        except Exception as e:
            print(f"错误: 调用响应模型 (GPT-4O) 失败: {e}")
            return "调用响应模型时出错。"
    # def _get_gpt_response(
    #     self, 
    #     question: str, 
    #     context: Dict,
    #     response_model: str,
    #     api_key: str,
    #     base_url: str,
    #     temperature: float,
    #     top_p: float
    # ) -> str:
    #     """使用GPT-4O生成对侦探询问的回复"""
    #     from arbench.utils.inference import inference
        
    #     # 构建GPT的系统提示
    #     gpt_system = """你是一个协助侦探破案的助手。侦探会询问你关于案件调查策略的建议。
    #     请根据案件背景提供简洁、有用的建议，帮助侦探更好地进行调查。
    #     你的回答应该：
    #     1. 简洁明了，1-2句话
    #     2. 基于案件事实和逻辑推理
    #     3. 提供具体可行的建议"""
        
    #     # 构建上下文信息
    #     case_info = ""
    #     if context:
    #         if 'case_info' in context:
    #             case_info = f"案件背景：{context['case_info']}\n"
    #         if 'suspect' in context:
    #             case_info += f"当前讨论的嫌疑人：{context['suspect']}\n"
    #         if 'turn' in context:
    #             case_info += f"当前是第{context['turn']}轮询问\n"
    #         if 'record' in context and context['record']:
    #             case_info += f"已有调查记录：{context['record'][:200]}...\n"
        
    #     gpt_messages = [
    #         {"role": "system", "content": gpt_system},
    #         {"role": "user", "content": f"{case_info}\n侦探的问题：{question}"}
    #     ]
        
    #     # 调用GPT-4O
    #     response = inference(
    #         gpt_messages, 
    #         model=response_model,
    #         temperature=temperature,
    #         top_p=top_p,
    #         api_key=api_key,
    #         base_url=base_url
    #     )
        
    #     return response.choices[0].message.content.strip()
    
    def _extract_final_answer(self, completion: str) -> str:
        """从completion中提取最终答案"""
        # 提取</think>之后的内容
        if "</think>" in completion:
            parts = completion.split("</think>")
            if len(parts) > 1:
                final_part = parts[1].strip()
                if final_part:
                    return final_part
        
        # 如果没有</think>之后的内容，尝试提取最后的有效内容
        lines = completion.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # 跳过标签和空行
            if line and not line.startswith('<') and not line.startswith('</'):
                # 这可能是最终答案
                return line
        
        # 默认返回一个简单答案
        return "无法确定"