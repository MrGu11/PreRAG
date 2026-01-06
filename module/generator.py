import json
import sys
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any, Callable, Tuple
import re

# 配置日志（基础配置，可在外部重写）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到模块搜索路径（便于导入内部模块）
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
from llm.llm import get_llm_answer  # 假设该模块已存在


class RAGGenerator:
    """RAG系统的生成模块，负责构建提示词并调用大模型生成回答

    修改点：judge_doc_efficient 支持输出 chain-of-thought（CoT）以提高判断准确率。
    新增参数：return_cot（默认 True）——如果为 True，函数返回 (bool, cot_text)，否则返回 bool（兼容旧用法）。
    """
    
    # 常量定义 - 集中管理配置参数（便于全局调整）
    DEFAULT_MODEL = "qwen3-14b"
    DEFAULT_MAX_CONTEXT_LENGTH = 4000
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_RETRY_TIMES = 2  # 统一重试次数配置

    # 默认指令模板（按场景分类，便于批量修改）
    DEFAULT_INSTRUCTIONS = {
        "subquery": (
            "You are a knowledgeable assistant. "
            "Please answer the query based on the provided documents. "
            "If the documents are insufficient or incomplete, you may use your own reasoning "
            "and general knowledge to produce the answer. "
            "Give the answer as much as possible."
        ),
        "final": (
            "You are a knowledgeable assistant. "
            "Infer from the answers to the subqueries the answer to the original query. "
            "There may be some subqueries that do not contribute to the final answer. "
            "Give the answer as much as possible. "
            "The answer only needs to answer the question and does not need to include reasoning process."
        ),
        "direct": "Please provide the final answer to the query. Output only the answer with no extra content."
    }
    
    # 提示词模板（统一管理所有模板，避免硬编码）
    PROMPT_TEMPLATES = {
        "without_retrieval": """{instruction}

query:
{query}

output:
final answer""",
        
        "subquery": """{instruction}

Context:
{context}

query:
{query}

Answer: (Output only the concise answer)""",
        
        "final": """{instruction}

Original Query:
{query}

Subquery Answers:
{subquery_section}

Final Answer: (Output only the concise answer)""",
        
        # 新增 CoT 评估模板：要求模型给出 step-by-step 的理由，并在最后一行给出明确标签
        "judge_doc_efficient_cot": """
You are a professional question-answering system evaluator. Your task is to determine whether given documents can answer a specific question.

Please follow these steps exactly:
1) Carefully read the question and documents.
2) Show your step-by-step reasoning (brief but explicit) about whether the documents contain enough information to directly and completely answer the question.
3) After your reasoning, on a single last line output the final binary decision in the exact format: "Final: Yes" or "Final: No" (without extra punctuation).

Rules:
* The reasoning can be a few short numbered or bullet points.
* The final decision must appear **only** on the last line and use the exact token pattern: Final: Yes  OR  Final: No

Input:
question: {query}
documents: {docs}
"""
    }

    def __init__(
        self,
        llm_call_func: Callable[[str, str], str] = get_llm_answer,
        model_id: str = DEFAULT_MODEL,
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        初始化RAG生成器
        """
        self.llm_call_func = llm_call_func
        # self.model_id = model_id
        self.model_id = 'qwen-max'
        self.max_context_length = max_context_length
        self.chunk_overlap = chunk_overlap
        
        # 验证核心依赖
        if not callable(llm_call_func):
            raise ValueError("llm_call_func must be a callable function")

    def _extract_document_text(self, doc: Dict[str, Any]) -> str:
        text_fields = ["text", "content", "context"]
        for field in text_fields:
            if field in doc:
                return str(doc[field]).strip()
        logger.warning("Document missing text fields (text/content/context)")
        return ""

    def _build_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No relevant documents provided."
        context_chunks = [
            f"Document {i}:\n{self._extract_document_text(doc)}"
            for i, doc in enumerate(docs, 1)
        ]
        return "\n\n".join(context_chunks)

    def create_direct_prompt(self, query: str, instruction: Optional[str] = None) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        instruction = instruction or self.DEFAULT_INSTRUCTIONS["direct"]
        return self.PROMPT_TEMPLATES["without_retrieval"].format(
            instruction=instruction,
            query=query.strip()
        )

    def create_subquery_prompt(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        instruction: Optional[str] = None
    ) -> str:
        if not query.strip():
            raise ValueError("Subquery cannot be empty")
        instruction = instruction or self.DEFAULT_INSTRUCTIONS["subquery"]
        context = self._build_context_from_docs(docs)
        return self.PROMPT_TEMPLATES["subquery"].format(
            instruction=instruction,
            context=context,
            query=query.strip()
        )

    def create_final_prompt(
        self,
        query: str,
        subquery_answers: Dict[str, str],
        instruction: Optional[str] = None
    ) -> str:
        if not query.strip():
            raise ValueError("Original query cannot be empty")
        if not subquery_answers:
            logger.warning("No subquery answers provided for final prompt")
        instruction = instruction or self.DEFAULT_INSTRUCTIONS["final"]
        subquery_section = "\n\n".join([
            f"Subquery {i}: {subq.strip()}\nAnswer: {ans.strip() or 'No answer'}"
            for i, (subq, ans) in enumerate(subquery_answers.items(), 1)
        ])
        return self.PROMPT_TEMPLATES["final"].format(
            instruction=instruction,
            query=query.strip(),
            subquery_section=subquery_section
        )

    def judge_doc_efficient(self, query: str, docs: List[str], return_cot: bool = False) -> Any:
        """
        判断文档是否能有效回答查询（直接且完整）。

        Args:
            query: 待判断的查询文本
            docs: 待评估的文档文本列表（list of strings）
            return_cot: 是否返回 chain-of-thought 文本；
                        如果 True，返回 (bool_decision, cot_text)
                        如果 False，返回 bool_decision（向后兼容）

        Returns:
            bool 或 (bool, cot_str)
        """
        # 确保 docs 是字符串形式的列表
        docs_text = json.dumps(docs, ensure_ascii=False)

        prompt = self.PROMPT_TEMPLATES["judge_doc_efficient_cot"].format(
            query=query,
            docs=docs_text
        )

        # 调用模型获取带推理的回答
        try:
            raw = self.generate(prompt, retry=self.DEFAULT_RETRY_TIMES)
        except Exception as e:
            logger.error(f"LLM call failed for judge_doc_efficient: {e}")
            return (False, "") if return_cot else False

        cot_text = str(raw).strip()

        # 解析最后一行是否包含 Final: Yes/No 标记
        final_match = re.search(r"final\s*[:\-]?\s*(yes|no)\s*$", cot_text, flags=re.IGNORECASE | re.MULTILINE)
        decision: Optional[bool] = None
        if final_match:
            label = final_match.group(1).lower()
            decision = True if label == "yes" else False
        else:
            # 退化策略：尝试从最后三行中寻找 yes/no
            last_lines = cot_text.strip().splitlines()[-3:]
            candidate = " ".join(last_lines).lower()
            if re.search(r"\bfinal\b.*\byes\b", candidate):
                decision = True
            elif re.search(r"\bfinal\b.*\bno\b", candidate):
                decision = False
            else:
                # 最后手段：检查文本中 yes/no 出现的次数和上下文——不可靠但可用作回退
                yes_count = len(re.findall(r"\byes\b", cot_text, flags=re.IGNORECASE))
                no_count = len(re.findall(r"\bno\b", cot_text, flags=re.IGNORECASE))
                if yes_count == 0 and no_count == 0:
                    decision = False  # 没有明确信号，保守判定为不可回答
                else:
                    decision = True if yes_count >= no_count else False

        if return_cot:
            return decision, cot_text
        else:
            return decision

    def generate(self, prompt: str, retry: int = DEFAULT_RETRY_TIMES) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        for attempt in range(retry + 1):
            try:
                logger.info(f"Calling LLM (model: {self.model_id}) - attempt {attempt + 1}")
                answer = self.llm_call_func(prompt, self.model_id)
                if answer is None:
                    raise ValueError("LLM returned empty response")
                return str(answer).strip()
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {str(e)}")
                if attempt == retry:
                    logger.error("All retry attempts exhausted")
                    raise
                continue

    def __repr__(self) -> str:
        return (
            f"RAGGenerator(model_id={self.model_id!r}, "
            f"max_context_length={self.max_context_length!r}, "
            f"chunk_overlap={self.chunk_overlap!r})"
        )


# 使用示例（仅在作为脚本运行时执行）
if __name__ == "__main__":
    generator = RAGGenerator(llm_call_func=get_llm_answer)
    query = "When did Abasingammedda leave the British empire?"
    docs = ["Abasingammedda is a village in central Sri Lanka. It is just a mile to the east of Kandy. It has a population of about 11,000."]

    decision, cot = generator.judge_doc_efficient(query, docs, return_cot=True)
    print("Decision:", decision)
    print("CoT:\n", cot)
