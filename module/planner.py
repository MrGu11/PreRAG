import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sys
import networkx as nx
from textwrap import dedent

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加上级目录到系统路径
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from llm.llm import get_llm_answer
from module.generator import RAGGenerator
from module.retriever import RAGRetriever


class RAGPlanner:
    """RAG系统的规划模块，负责查询分解、子查询依赖管理和执行规划"""
    
    def __init__(self, 
                 llm_call_func: callable = get_llm_answer,
                 memory_path: str = "../data/DAG.json",
                 model_id: str = "qwen3-14b",
                 max_context_length: int = 4000,
                 chunk_overlap: int = 100):
        """
        初始化RAG规划器
        
        Args:
            llm_call_func: 调用大模型的函数，应接受messages参数并返回回复
            memory_path: DAG记忆存储地址
            model_id: 使用的模型ID
            max_context_length: 最大上下文长度
            chunk_overlap: 上下文块之间的重叠长度
        """
        self.global_dag = nx.DiGraph()  # 全局DAG图，管理查询依赖关系
        self.llm_call_func = llm_call_func
        self.max_context_length = max_context_length
        self.chunk_overlap = chunk_overlap
        self.memory_path = memory_path
        self.model_id = model_id
        self.subquery_embeddings = {}  # 子查询的嵌入存储

        # # 加载历史记忆
        # self.load_memory()

    def add_original_query(self, query: str) -> None:
        """添加原始查询作为DAG的初始节点"""
        if query not in self.global_dag.nodes:
            self.global_dag.add_node(query, completed=False)
            logger.warning("已添加初始查询到DAG图")
        else:
            logger.warning(f"查询 '{query}' 已存在于DAG中")

    def load_memory(self) -> None:
        """从JSON文件加载历史查询记忆和DAG结构"""
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                
            # 重建节点
            for node_data in memory.get('nodes', []):
                subquery = node_data['subquery']
                attrs = node_data.get('attrs', {})
                self.global_dag.add_node(subquery, **attrs)
            
            # 重建边
            for edge in memory.get('edges', []):
                self.global_dag.add_edge(
                    edge['source'], 
                    edge['target'], 
                    **edge.get('attrs', {})
                )
            
            # 加载嵌入
            self.subquery_embeddings = memory.get('embeddings', {})
            logger.warning(f"成功从 {self.memory_path} 加载记忆，节点数: {len(self.global_dag.nodes)}")
            
        except FileNotFoundError:
            logger.warning(f"未找到记忆文件 {self.memory_path}，将初始化新的DAG")
            self.global_dag = nx.DiGraph()
            self.subquery_embeddings = {}
        except json.JSONDecodeError:
            logger.error(f"记忆文件 {self.memory_path} 格式错误，无法解析")
            self.global_dag = nx.DiGraph()
            self.subquery_embeddings = {}
        except Exception as e:
            logger.error(f"加载记忆时发生错误: {str(e)}")
            self.global_dag = nx.DiGraph()
            self.subquery_embeddings = {}

    def save_memory(self) -> None:
        """将当前DAG状态保存到JSON文件"""
        try:
            # 准备节点数据
            nodes_data = [
                {'subquery': node, 'attrs': dict(self.global_dag.nodes[node])}
                for node in self.global_dag.nodes
            ]
            
            # 准备边数据
            edges_data = [
                {'source': u, 'target': v, 'attrs': dict(self.global_dag.edges[u, v])}
                for u, v in self.global_dag.edges
            ]
            
            # 构建完整记忆数据
            memory = {
                'nodes': nodes_data,
                'edges': edges_data,
                'embeddings': self.subquery_embeddings
            }
            
            # 确保目录存在
            Path(self.memory_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            logger.warning(f"已将DAG状态保存到 {self.memory_path}")
            
        except Exception as e:
            logger.error(f"保存记忆时发生错误: {str(e)}")

    def merge_dag(self, sub_dag: nx.DiGraph, original_query: str) -> None:
        """
        将子查询DAG合并到全局DAG中
        
        Args:
            sub_dag: 子查询DAG图
            original_query: 原始查询（将被子DAG替代的节点）
        """
        if original_query not in self.global_dag.nodes:
            logger.warning(f"原始查询 '{original_query}' 不在全局DAG中，无法合并")
            return

        # 合并子DAG到全局DAG
        self.global_dag = nx.compose(self.global_dag, sub_dag)
        logger.warning(f"合并子DAG，新增节点数: {len(sub_dag.nodes)}")

        # 1. 获取原始查询在全局DAG中的前驱节点
        global_predecessors = list(self.global_dag.predecessors(original_query))
        
        # 2. 获取子DAG中的根节点（无前驱的节点）
        subdag_roots = [
            node for node in sub_dag.nodes 
            if len(list(sub_dag.predecessors(node))) == 0
        ]
        
        # 3. 连接全局前驱到子DAG根节点
        for pred in global_predecessors:
            for root in subdag_roots:
                self.global_dag.add_edge(pred, root)
                logger.debug(f"添加边: {pred} -> {root}")

        # 4. 获取原始查询在全局DAG中的后继节点
        global_successors = list(self.global_dag.successors(original_query))
        
        # 5. 获取子DAG中的叶子节点（无后继的节点）
        subdag_leaves = [
            node for node in sub_dag.nodes 
            if len(list(sub_dag.successors(node))) == 0
        ]
        
        # 6. 连接子DAG叶子节点到全局后继
        for leaf in subdag_leaves:
            for succ in global_successors:
                self.global_dag.add_edge(leaf, succ)
                logger.warning(f"添加边: {leaf} -> {succ}")

        # 移除原始查询节点及其相关边
        # self.global_dag.remove_node(original_query)
        logger.warning(f"已移除原始查询节点: {original_query}")

        # 保存更新后的DAG
        self.save_memory()

    def get_searchable_subquery_prompt(self,  
                                      subquery: str, 
                                      dependencies: List[str], 
                                      dependency_answers: List[str]) -> str:
        """
        生成用于将子查询转换为可搜索查询的提示词
        
        Args:
            subquery: 目标子查询
            dependencies: 依赖的子查询列表
            dependency_answers: 依赖子查询的答案列表
        
        Returns:
            构建好的提示词
        """

        dependent_lines = [
            f"- Subquery: {q or 'N/A'}\n  Answer: {a or 'N/A'}" 
            for q, a in zip(dependencies, dependency_answers)
        ]
        dependent_info = "\n".join(dependent_lines) if dependent_lines else "No dependent subqueries."

        return(f"""You are a professional query converter. Your task is to generate a complete, search-ready natural language query based on a given target subquery and dependency information.

**Conversion Steps:**
1. Identify the core question in the target subquery.
2. Extract the key answer from the dependency information.
3. Embed the answer into the core question to generate a coherent natural language query.
4. The output query needs to be a single fact query.

**Requirements:**
- The final output must be a standalone, complete, searchable query.
- Output only the query itself, without any explanation or additional text.

**Example:**
**Input:**
[Target Subquery]: "[Green >> performer] >> spouse"
[Dependent Information]:
- Subquery: who is the performer of Green?
Answer: Steve Hillage
**Output:**
Who is Steve Hillage's spouse?

---
**Now start the transformation:**
[Target Subquery]
{subquery}

[Dependent Information]
{dependent_info}

[Searchable Query]
""")

    def get_searchable_subquery_again_prompt(self,  
                                        subquery: str) -> str:
        """
        生成用于将子查询转换为可搜索查询的提示词
        
        Args:
            original_query: 原始查询
            subquery: 目标子查询
        
        Returns:
            构建好的提示词
        """

        return(f"""You are a professional query converter. Your task is to rewrite a complete natural language query based on a given target subquery.

**Requirements:**
- The final output must be a standalone, complete, searchable query.
- The output query needs to be a single fact query.
- Output only the query itself, without any explanation or additional text.

**Now start the transformation:**
[Target Subquery]
{subquery}

[rewrite Query]
""")


    def select_next_subqueries(self, retriever: RAGRetriever, original_query: str, answer_dict: Dict[str, str], docs: List[str]) -> List[str]:
        """
        选择所有前驱都已完成的子查询，准备执行
        
        Args:
            original_query: 原始查询
            answer_dict: 已完成子查询的答案字典
        
        Returns:
            可执行的子查询列表
        """
        ready_subqueries = []
        generator = RAGGenerator(llm_call_func=get_llm_answer, model_id=self.model_id)
        
        for subquery in list(self.global_dag.nodes):  # 使用列表避免迭代中修改的问题
            # 检查节点是否已完成
            if self.global_dag.nodes[subquery].get('completed', False):
                continue
            
            # 获取所有前驱节点
            predecessors = list(self.global_dag.predecessors(subquery))
            
            # 检查所有前驱是否都已完成
            if predecessors == [] or all(self.global_dag.nodes[pred].get('completed', False) for pred in predecessors):
                # 生成可搜索的子查询
                try:
                    # 生成提示词并调用LLM
                    pred_answers = [answer_dict[pred] for pred in predecessors]

                    prompt = self.get_searchable_subquery_prompt(
                        subquery, 
                        predecessors, 
                        pred_answers
                    )
                    searchable_query = self.llm_call_func(prompt, self.model_id).strip()
                    
                    # 重命名节点
                    if searchable_query and searchable_query != subquery:
                        self.global_dag = nx.relabel_nodes(self.global_dag, {subquery: searchable_query})
                        logger.warning(f"子查询重命名: {subquery} -> {searchable_query}")
                        subquery = searchable_query
                    
                    ready_subqueries.append(subquery)
                    logger.warning(f"添加可执行子查询: {subquery}")
                
                except Exception as e:
                    logger.error(f"处理子查询 '{subquery}' 时出错: {str(e)}")
                    continue
        
        logger.warning(f"找到 {len(ready_subqueries)} 个可执行子查询")
        return ready_subqueries

    def generate_subqueries_with_anchors(self, query: str, retrieval_anchors: List[List[str]]) -> nx.DiGraph:
        """
        根据检索锚点和原始查询生成子查询DAG
        
        Args:
            query: 原始查询
            retrieval_anchors: 检索锚点列表
        
        Returns:
            子查询DAG图
        """
        # 构建提示词
        prompt = f"""System / Instruction:
You are a professional information retrieval assistant specialized in splitting user queries into independent, fact-answerable subqueries. ALWAYS output exactly one valid JSON object and nothing else. Use deterministic output (temperature 0). Keep length concise.

User prompt:
Inputs:
- original_query: {query}
- retrieval_anchors: {retrieval_anchors}   // JSON array of strings

Important definitions:
- retrieval_anchor: a keyword set typically found within a single article; may be misleading.
- valid subquery: short interrogative/imperative phrase that can be answered by a single article with a concrete fact/data/statement.
- simple factual question: a single atomic question requiring one fact (e.g., "When was X born?", "What is the population of Y?").

Rules (must follow exactly):
1) Split rules:
   - Use retrieval anchors as preferred split boundaries.
   - Each subquery must be answerable by a single article and produce a clear factual conclusion (no open-ended "how to" / "discuss the impact" / "analyze" questions).
   - Subqueries should be concise (recommended length 4-25 words), minimally overlapping, and collectively sufficient to reconstruct the original query intent.
   - You are allowed to reference the output of other subqueries within a subquery using square brackets [...] (see dependency rules and examples).

2) Dependency rules:
   - Output dependencies as an array of pairs [child_subquery_text, parent_subquery_text].
   - Each pair means: child requires parent answer to be answerable.
   - Parent and child must appear exactly in "subqueries".
   - No self-dependency and no cycles allowed.

3) Anchors mapping:
   - For each subquery include which retrieval_anchors it uses (anchors_used).
   - Provide a confidence score per subquery in [0.0,1.0] indicating how well the subquery maps to anchors and whether it is likely answerable by a single article.
   - For a single retrieval anchor, multiple related atomic subqueries can be generated.

4) No-split condition:
   - If the original query is a simple factual question, return it as the single subquery and set dependencies to [].

5) Validation:
   - Ensure dependencies reference existing subqueries.
   - If any rule cannot be satisfied (e.g., cycle would be required, or cannot create fact-answerable subqueries), return JSON with "error" and a short "reason".

Output JSON schema (exact keys):
{{
  "query": "<original_query>",
  "subqueries": [
    {{
      "text": "<subquery_text>",
      "anchors_used": ["anchor1","anchor2",...],
      "confidence": 0.0-1.0
    }},
    ...
  ],
  "dependencies": [
    ["<child_subquery_text>","<parent_subquery_text>"],
    ...
  ],
  "CoT": "<one-sentence concise rationale>",
  "validation": {{"ok": true}}   // or {{"ok": false, "error":"reason"}}
}}

Additional guidance:
- When a retrieval_anchor is misleading, prefer splitting such that misleading parts become explicit subqueries to be verified (e.g., "Does anchor X actually refer to Y?").
- If a node must be shared by multiple subqueries, allow it but prefer creating a small "context" subquery that other subqueries depend on.
- Keep CoT to 1 sentence describing splitting logic (e.g., "Split by anchors A,B; P2 depends on P1 because it references the resolved entity.").

Example output:
{{
  "query":"When did the national lottery start in the country where the London museum prominently located in Trafalgar Square is found?",
  "subqueries":[
    {{"text":"Which London museum is prominently located in Trafalgar Square?", "anchors_used":["Trafalgar Square","museum"],"confidence":0.98}},
    {{"text":"[Which London museum is prominently located in Trafalgar Square?] >> country", "anchors_used":["country"],"confidence":0.90}},
    {{"text":"when did the national lottery start in [[Which London museum is prominently located in Trafalgar Square?] >> country]", "anchors_used":["the national lottery"],"confidence":0.90}}
  ],
  "dependencies":[
    ["[Which London museum is prominently located in Trafalgar Square?] >> country", "Which London museum is prominently located in Trafalgar Square?"],
    ["when did the national lottery start in [[Which London museum is prominently located in Trafalgar Square?] >> country]", "[Which London museum is prominently located in Trafalgar Square?] >> country"]
  ],
  "CoT":"First resolve which performer is meant by 'Green', then query that performer's spouse.",
  "validation":{{"ok":true}}
}}

Now perform the split and return the single JSON object.
"""

        
        # 调用LLM生成子查询计划
        count = 0
        while(count < 3):
            try:
                answer = self.llm_call_func(prompt, self.model_id)
                logger.warning(f"LLM返回的子查询计划: {answer}")
                
                # 解析LLM输出
                plan = self._parse_llm_json_output(answer)
                
                # 构建DAG图
                subquery_dag = nx.DiGraph()
                
                # 添加节点
                for subquery in plan['subqueries']:
                    subquery_dag.add_node(subquery, completed=False)
                
                # 添加边（依赖关系）
                for child, parent in plan['dependencies']:
                    if child != parent and child in plan['subqueries'] and parent in plan['subqueries']:
                        subquery_dag.add_edge(parent, child)
                    else:
                        logger.warning(f"无效的依赖关系: {[child, parent]}，已跳过")
                        continue
                if not nx.is_directed_acyclic_graph(subquery_dag):
                    count += 1
                    logger.warning(f"存在环，已跳过")
                    continue
                break_flag = False
                for subquery in subquery_dag.nodes:
                    if subquery in self.global_dag.nodes and subquery != query:
                        count += 1
                        logger.warning(f"当前subquery已出现过，重新进行拆分")
                        break_flag = True
                        continue
                if break_flag == True:
                    continue
                logger.warning(f"生成带锚点的子查询DAG，节点数: {len(subquery_dag.nodes)}，边数: {len(subquery_dag.edges)}")
                return subquery_dag
            except Exception as e:
                count += 1
                logger.error(f"生成带锚点的子查询时出错: {str(e)}")
        return nx.DiGraph()

    def generate_subqueries_without_anchors(self, query: str) -> List[str]:
        """
        无检索锚点时生成子查询列表
        
        Args:
            query: 原始查询
        
        Returns:
            子查询列表
        """
        prompt = dedent(f"""
        You are a search-aware query understanding expert who is good at splitting complex queries into several subqueries suitable for the search system to process. Now, please perform semantic analysis and intention decomposition on the complex query input by the user, and output a structured subquery set.

        Task goal: Split the input complex query so that each subquery:
        1. Semantically complete
        2. Closely related to the original information needs
        3. Suitable for direct document retrieval

        Input:
        Original query: <{query}>

        Output format:(json)
        {{"query": original_query, "subqueries": [subquery1, subquery2, ...]}}
        """).strip()
        
        try:
            answer = self.llm_call_func(prompt, self.model_id)
            plan = self._parse_llm_json_output(answer)
            subqueries = plan.get('subqueries', [])
            logger.warning(f"生成无锚点的子查询，数量: {len(subqueries)}")
            return subqueries
        except Exception as e:
            logger.error(f"生成无锚点的子查询时出错: {str(e)}")
            return []

    @staticmethod
    def _parse_llm_json_output(llm_output: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON格式输出
        
        Args:
            llm_output: LLM的原始输出
        
        Returns:
            解析后的字典
        """
        # 清理输出中的多余标记
        cleaned_output = llm_output.replace('```', '').replace('json', '').strip()
        
        # 尝试解析JSON
        try:
            item = json.loads(cleaned_output)
            if isinstance(item['subqueries'][0], dict):
                item['subqueries'] = [sub['text'] for sub in item.get('subqueries', [])]
            return item
        except json.JSONDecodeError as e:
            logger.error(f"解析LLM输出为JSON时失败: {str(e)}. 原始输出: {cleaned_output}")
            raise

    


# 使用示例
if __name__ == "__main__":
    # 示例查询和子查询
    query = "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?"
    retrieval_anchors = [["Aruba", "continent"], ["Prazeres"]]

    # 初始化生成器
    planner= RAGPlanner()
    
    # 初始化DAG图
    planner.add_original_query(query)

    # 保存DAG图
    planner.save_memory()

    # 生成subqueries
    # 可能出现的错误：ValueError: None cannot be a node
    subquery_dag = planner.generate_subqueries(query, retrieval_anchors)

    # 合并DAG
    planner.merge_dag(subquery_dag, query)

    # 保存
    planner.save_memory()

    # 输出无前驱节点
    next_subqueries = planner.select_next_subqueries()
    # print(next_subqueries)

    # # 输出可视化DAG图
    # planner.visualize_dag_matplotlib(output_path="/data0/thgu/query-decomposition/RAG/data/DAG.png")