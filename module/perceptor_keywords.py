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
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

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
            examples=["""Example 1:
Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
],
            query=query
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
            return {}
        
        keywords_dict = json.loads(response.strip())
        keywords_dict['keywords'] = []
        for key_type in ["high_level_keywords", "low_level_keywords"]:
            if key_type in keywords_dict and isinstance(keywords_dict[key_type], list):
                keywords_dict['keywords'].extend(keywords_dict[key_type])
        return keywords_dict

    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        核心方法：从查询中提取关键词（命名实体）
        
        Args:
            query: 用户查询文本
        
        Returns:
            包含关键词列表的字典，格式为 {"keywords": [str, ...]}
        """
        for _ in range(3):  # 最多重试3次
            try:
                # 1. 构建提示词
                prompt = self._build_extraction_prompt(query)
                
                # 2. 调用大模型
                try:
                    logger.info(f"Calling LLM for keyword extraction (model: {self.model_id})")
                    response = self.llm_call_func(prompt, self.model_id)
                except Exception as e:
                    logger.error(f"LLM call failed during keyword extraction: {str(e)}")
                    return {}
                
                # 3. 解析结果并返回
                keywords_dict = self._parse_llm_response(response)
                logger.debug(f"Extracted keywords from query: {keywords_dict}")
                return keywords_dict
        
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                continue  # 重试
            except Exception as e:
                logger.error(f"Error during keyword extraction: {str(e)}")
                continue  # 重试
        return {} # 最终失败返回空列表
        

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