import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable

# 配置日志（统一日志输出，替代print警告）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目路径配置（简化路径引用，提升可读性）
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

# 导入依赖模块
from llm.llm import get_llm_answer


class RAGPerceptor:
    """
    RAG感知器：从用户查询中提取关键词（命名实体）和检索锚点，支撑后续检索流程
    """
    # 类级常量（集中管理配置，便于全局调整）
    DEFAULT_MODEL_ID = "qwen3-14b"
    DEFAULT_KEYWORD_EXTRACTION_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

3.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

4.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

    def __init__(
        self,
        llm_call_func: Callable[[str, str], str] = get_llm_answer,
        model_id: str = DEFAULT_MODEL_ID,
        custom_extraction_prompt: Optional[str] = None
    ):
        """
        初始化RAG感知器
        
        Args:
            llm_call_func: 大模型调用函数，需满足 (prompt: str, model_id: str) -> str 接口
            model_id: 调用的大模型ID（默认使用类配置的DEFAULT_MODEL_ID）
            custom_extraction_prompt: 自定义关键词提取提示词（可选，优先级高于默认模板）
        
        Raises:
            ValueError: 当llm_call_func不可调用时
        """
        # 验证核心依赖
        if not callable(llm_call_func):
            raise ValueError("llm_call_func must be a callable function that accepts (prompt, model_id)")
        
        self.llm_call_func = llm_call_func
        self.model_id = model_id
        self.extraction_prompt_template = custom_extraction_prompt or self.DEFAULT_KEYWORD_EXTRACTION_PROMPT

    def _build_extraction_prompt(self, query: str) -> str:
        """
        构建关键词提取提示词（填充查询到模板中）
        
        Args:
            query: 用户原始查询文本
        
        Returns:
            格式化后的完整提示词
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.extraction_prompt_template.format(
            entity_types="Person, Location, Organization, Event, Product, Other",
            tuple_delimiter=" | ",
            language="English",
            examples=(
                "entity | Tokyo | Location | Tokyo is the capital city of Japan.\n"
                "entity | Albert Einstein | Person | Albert Einstein was a theoretical physicist known for the theory of relativity."
            ),
            input_text=query
        )

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        解析大模型返回结果，提取关键词列表（增强鲁棒性）
        
        Args:
            response: 大模型返回的原始字符串
        
        Returns:
            提取到的关键词列表（解析失败返回空列表）
        """
        if not response.strip():
            logger.warning("LLM returned empty response for keyword extraction")
            return []
        
        # 按行分割并提取关键词
        keywords = []
        for line in response.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 4 and parts[0].lower() == "entity":
                entity_name = parts[1]
                keywords.append(entity_name)
        return keywords

    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        核心方法：从查询中提取关键词（命名实体）
        
        Args:
            query: 用户查询文本
        
        Returns:
            包含关键词列表的字典，格式为 {"keywords": [str, ...]}
        """
        # 1. 构建提示词
        prompt = self._build_extraction_prompt(query)
        
        # 2. 调用大模型
        try:
            logger.info(f"Calling LLM for keyword extraction (model: {self.model_id})")
            response = self.llm_call_func(prompt, self.model_id)
        except Exception as e:
            logger.error(f"LLM call failed during keyword extraction: {str(e)}")
            return {"keywords": []}
        
        # 3. 解析结果并返回
        keywords = self._parse_llm_response(response)
        logger.debug(f"Extracted keywords from query: {keywords}")
        return {"keywords": keywords}

    def __repr__(self) -> str:
        """对象字符串表示（便于调试和日志输出）"""
        return (
            f"RAGPerceptor(model_id={self.model_id!r}, "
            f"uses_custom_prompt={self.extraction_prompt_template != self.DEFAULT_KEYWORD_EXTRACTION_PROMPT!r})"
        )
        
    
if __name__ == "__main__":

    query = "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?"

    # 建立Perceptor实例
    perceptor = RAGPerceptor()
    
    # 提取query关键词
    item = perceptor.extract(query)

    print(item)