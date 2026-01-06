from tqdm import tqdm
import json
import random
from openai import OpenAI
# from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
# from PIL import ImageFile
import base64
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import time
try:
    from nltk.stem import SnowballStemmer
except Exception:
    # Fallback dummy stemmer if nltk is not installed in the environment
    class SnowballStemmer:
        def __init__(self, *args, **kwargs):
            pass
        def stem(self, word):
            return word
from http import HTTPStatus
import dashscope
import re
import argparse

local_url = "http://127.0.0.1:8020/v1"

client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key='your_api_key',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # base_url=local_url,
    )


def gen_decomposition_with_anchor_prompt(query, anchors):
    prompt = f"""You are a professional information retrieval assistant. Please process user queries according to the following requirements:

[Original query]

{query}

[Retrieval anchor]

{anchors}


[Task requirements]

1. The subqueries must maintain a logical association to ensure that the splitting of the subqueries does not distort the semantics of the original query.

2. Refer to the retrieval anchor as the segmentation boundary and split the original query into multiple subqueries (the retrieval anchor may contain misleading keywords).

3. Describe the dependency relationship between subqueries (for example: "Subquery 2 depends on the result of subquery 1").

4. Some subqueries cannot give their retrieval anchors because their predecessors are not completed.

【Output format】(json)
{{"query": original_query, "subquery_list": subquery_list}}
"""
    return prompt

def gen_direct_decomposition_prompt(query):
    prompt = f"""You are a search-aware query understanding expert who is good at splitting complex queries into several subqueries suitable for the search system to process. Now, please perform semantic analysis and intention decomposition on the complex query input by the user, and output a structured subquery set.

Task goal: Split the input complex query so that each subquery:

1. Semantically complete

2. Closely related to the original information needs

3. Suitable for direct document retrieval

Input:

Original query: <{query}>

Output format:(json)
{{"query": original_query, "subquery_list": subquery_list}}
"""
    
    return prompt

def get_searchable_subquery_prompt(subquery, subqueries, answers):
    prompt = (
        "Now given a subquery and its dependent subqueries and their answers,"
        "where the elements of answers correspond one-to-one to the elements of subqueries,"
        "you are now required to generate a complete and searchable query statement for the specified subquery based on the subqueries and their answers."
        "\n\n"
    )
    prompt += "【Input】\n"
    prompt += f"subquery: {subquery}\n"
    prompt += f"dependent subqueries: {subqueries}"
    prompt += f"dependent subqueries' answer: {answers}\n\n"
    prompt += "【Output】\n"
    prompt += "searchable subquery"


    return prompt

def get_llm_answer(
    prompt,
    model_id,
    thinking_mode: str = "fast",  # "slow" | "fast"
    temperature: float = 0.0
):
    """
    thinking_mode:
        - "slow": 启用满思考（reasoning_content）
        - "fast": 关闭思考，直接生成答案
    """

    thinking_chunks = []
    answer_chunks = []

    enable_thinking = thinking_mode == "slow"

    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
        extra_body={
            "enable_thinking": enable_thinking
        }
    )

    for chunk in stream:
        delta = chunk.choices[0].delta

        # 🧠 慢思考 only
        if enable_thinking and delta.reasoning_content:
            thinking_chunks.append(delta.reasoning_content)

        # ✅ 最终答案（两种模式都有）
        if delta.content:
            answer_chunks.append(delta.content)

    answer = "".join(answer_chunks)

    thinking = "".join(thinking_chunks)

    return answer
    # return answer.split('</think>')[1].strip()



if __name__ == "__main__":
    answer = get_llm_answer("你是谁", 'qwen3-14b')
    print(answer)  # =1 表示完全相同