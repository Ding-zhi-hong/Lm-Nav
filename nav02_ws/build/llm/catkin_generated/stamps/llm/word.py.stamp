#!/usr/bin/env python3
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import re

class WordSplitter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, legacy=True)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def split_sentence(self, sentence):
        """
        将句子拆分为单个词语
        返回词语列表
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未正确加载")
        
        # 设计提示词
        prompt = f"""
        Task: Split the following sentence into individual words.
        Rules:
        - Output comma-separated words
        - Preserve original case
        - Include punctuation as separate words
        - Do not add any explanations
        
        Example:
        Input: "Hello, how are you?"
        Output: "Hello, , how, are, you, ?"
        
        Now split:
        Input: "{sentence}"
        Output:"""
        
        try:
            # 编码输入
            input_ids = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    length_penalty=0.8
                )
            
            # 解码输出
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 处理输出
            return self.process_output(decoded)
        
        except Exception as e:
            print(f"词语拆分失败: {str(e)}")
            return []
    
    def process_output(self, raw_output):
        """
        处理模型输出，提取词语列表
        """
        # 尝试找到输出部分
        if "Output:" in raw_output:
            # 提取输出部分
            output_part = raw_output.split("Output:")[1].strip()
        elif "Words:" in raw_output:
            output_part = raw_output.split("Words:")[1].strip()
        else:
            output_part = raw_output
        
        # 分割词语
        words = []
        if ',' in output_part:
            words = [word.strip() for word in output_part.split(',')]
        else:
            # 尝试其他分隔符
            words = re.split(r'\s+', output_part)
        
        # 清理词语
        cleaned_words = []
        for word in words:
            # 移除多余空格
            word = word.strip()
            if word:
                # 处理特殊字符
                if word in [",", ".", "!", "?", ";", ":", "'", "\""]:
                    cleaned_words.append(word)
                else:
                    # 移除词语开头结尾的特殊字符
                    cleaned_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
                    if cleaned_word:
                        cleaned_words.append(cleaned_word)
        
        return cleaned_words

if __name__ == "__main__":
    # 测试功能
    MODEL_PATH = "/home/rosnoetic/nav02_ws/src/flan-t5-base"
    splitter = WordSplitter(MODEL_PATH)
    
    test_sentence = "Hello, how are you today?"
    words = splitter.split_sentence(test_sentence)
    print(f"原句: '{test_sentence}'")
    print(f"拆分结果: {words}")