# interactive_policy_client.py
import re
from typing import Dict, List, Optional, Tuple
import requests
import json
import time

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
        self.timeout = timeout
        self.stop_reason = None
        self.completion = "<think>\n"
        self.completion_tokens = max_tokens
        
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
        """
        生成带交互的回答
        返回: (最终答案, 完整completion, 交互历史, input_tokens, output_tokens)
        """
        
        # 重置状态
        self.completion = "<think>\n"
        self.completion_tokens = 4096
        self.stop_reason = None
        
        interaction_history = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 构建初始消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": self.completion}
        ]
        
        # 调用模型
        model_response = self._chat(messages)
        if model_response is None:
            return user_prompt, self.completion, [], 0, 0
            
        # 更新token计数
        usage = model_response.get('usage', {})
        total_input_tokens += usage.get('prompt_tokens', 0)
        total_output_tokens += usage.get('completion_tokens', 0)
        
        # 交互循环 - 跟你的代码一样简单
        while self.stop_reason == "</asking>":
            # 提取asking内容
            ask_match = re.search(r"<asking>(.*?)</asking>", self.completion, re.DOTALL)
            if not ask_match:
                break
                
            ask_content = ask_match.group(1).strip()
            
            # 调用GPT-4O获取回复
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
            
            # 添加response继续生成
            self.completion += f"\n<response>{gpt_response}</response>\n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": self.completion}
            ]
            
            # 继续调用模型
            model_response = self._chat(messages)
            if model_response is None:
                break
                
            # 更新token计数
            usage = model_response.get('usage', {})
            total_input_tokens += usage.get('prompt_tokens', 0)
            total_output_tokens += usage.get('completion_tokens', 0)
        
        # 确保completion以</think>结尾
        if "</think>" not in self.completion:
            self.completion += "\n</think>\n"
        
        # 提取最终答案
        final_answer = self._extract_final_answer(self.completion, user_prompt)
        
        return final_answer, self.completion, interaction_history, total_input_tokens, total_output_tokens
    
    def _chat(self, messages):
        """发送聊天请求 - 基于你的ModelClient.chat()"""
         # 调用distill模型
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'none'}"
        }
        
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": ["</asking>", "<｜end▁of▁sentence｜>"],
            "max_tokens": self.completion_tokens,
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
            
            if response.status_code == 200:
                results = response.json()
                
                # 更新状态 - 跟你的代码一样
                self.stop_reason = results['choices'][0].get('stop_reason', None)
                new_content = results['choices'][0]['message']['content']
                self.completion += new_content
                
                # 更新剩余tokens
                completion_tokens_used = results['usage']['completion_tokens']
                self.completion_tokens -= completion_tokens_used
                self.completion_tokens = max(1, self.completion_tokens)
                
                return results
            else:
                print(f"HTTP Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling model: {e}")
            return None
    
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
        """使用GPT-4O生成回复"""
        from arbench.utils.inference import inference
        
        # 简单的系统提示
        gpt_system = """You are an assistant who helps detectives solve cases. Please provide concise and useful suggestions based on the questions."""
        
        # 构建消息
        gpt_messages = [
            {"role": "system", "content": gpt_system},
            {"role": "user", "content": f"The detective's question:{question}"}
        ]
        
        try:
            response = inference(
                gpt_messages, 
                model=response_model,
                temperature=temperature,
                top_p=top_p,
                api_key=api_key,
                base_url=base_url
            )
            
            if response and response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            else:
                return "Unable to generate a response"
                
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return "Call failed"
    
    def _extract_final_answer(self, completion: str, user_prompt: str = "") -> str:
        """Extract the final answer from "completion""""
        
        # 首先尝试提取</think>之后的内容
        if "</think>" in completion:
            parts = completion.split("</think>")
            if len(parts) > 1:
                final_part = parts[1].strip()
                if final_part:
                    return final_part
        
        # 如果没有，从think内容中提取最后一个有意义的内容
        if "<think>" in completion:
            think_content = completion.split("<think>")[1]
            if "</think>" in think_content:
                think_content = think_content.split("</think>")[0]
            
            # 根据prompt类型智能提取
            if "choose a suspect" in user_prompt.lower():
                # 查找人名模式
                name_pattern = r'((?:Mr\.|Ms\.|Dr\.|Professor|Mrs\.) [A-Z][a-z]+ [A-Z][a-z]+)'
                matches = re.findall(name_pattern, think_content)
                if matches:
                    return matches[-1]  # 返回最后提到的名字
            
            elif "give your question" in user_prompt.lower():
                # 查找问句
                questions = re.findall(r'[A-Z][^?]*\?', think_content)
                if questions:
                    return questions[-1].strip()
            
            elif "true murderer" in user_prompt.lower():
                # 查找答案
                answer_match = re.search(r'\b([A-E])\b', think_content)
                if answer_match:
                    return answer_match.group(1)
        
        # 默认返回
        return completion.strip().split('\n')[-1] if completion.strip() else "unidentified"