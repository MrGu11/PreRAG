import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import networkx as nx
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - RAGSystem - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 导入项目模块
from module.perceptor import RAGPerceptor
from module.pre_retriever import RAGPreretriever
from module.generator import RAGGenerator
from module.retriever import RAGRetriever
from module.planner import RAGPlanner
from llm.llm import get_llm_answer


class RAGConfig:
    """RAG系统配置类，集中管理路径和参数"""
    # 系统参数
    MAX_RETRY = 3  # 最大重试次数
    LLM_MODEL_ID = "qwen3-14b"  # 默认LLM模型
    MAX_RECURSION_DEPTH = 2  # 最大递归切分深度，防止无限递归
    SHARE_MODULES_USE_LOCK = False
    DATASET_NAME = "hotpot"  # 数据集名称变量，可根据实际数据集修改
    TOP_N = 3  # 默认检索文档数
    
    # 数据路径（使用DATASET_NAME变量替换硬编码的hotpot）
    TEST_DATA_PATH = ROOT_DIR / "data" / f"test_data_mult_{DATASET_NAME}.json"
    OUTPUT_PATH = ROOT_DIR / "data" / "outputs" / f"test_data_preds_with_preretriever_{DATASET_NAME}_{LLM_MODEL_ID}_depth_{MAX_RECURSION_DEPTH}.jsonl"
    DETAIL_LOG_DIR = ROOT_DIR / "data" / "detail_log" / f"detail_log_depth_{DATASET_NAME}_{LLM_MODEL_ID}_{MAX_RECURSION_DEPTH}"  # 新增：详细日志路径
    EMBEDDING_CORPUS_DIR = ROOT_DIR / "data" / "embedding" / f"{DATASET_NAME}_corpus_embeddings"  # 新增：嵌入向量存储路径
    


def load_test_data(data_path: Path) -> Tuple[List[str], List[str], List[Any]]:
    """加载测试数据"""
    queries = []
    answers = []
    
    try:
        with open(data_path, "r", encoding='utf-8') as infile:
            item = json.load(infile)
            queries = (item['queries'])
            answers = (item['answers'])
            corpus = (item['corpus'])
        
        logger.warning(f"成功加载测试数据，共 {len(queries)} 条记录")
        return queries, answers, corpus
        
    except Exception as e:
        logger.error(f"加载测试数据失败: {str(e)}", exc_info=True)
        raise


def process_subquery_recursive(
    subquery: str,
    corpus: Any,
    perceptor: RAGPerceptor,
    planner: RAGPlanner,
    generator: RAGGenerator,
    retriever: RAGRetriever,
    pre_retriever: RAGPreretriever,
    answer_dict: Dict[str, str],
    detail_log: Dict[str, Any]  # 新增：用于记录详细信息的字典
) -> str:
    """
    递归处理子查询：检查覆盖率，不达标则进一步切分，达标则检索
    
    Args:
        subquery: 要处理的子查询
        corpus: 语料库
        perceptor: 洞察器实例
        planner: 规划器实例
        generator: 生成器实例
        retriever: 检索器实例
        pre_retriever: 预处理检索器实例
        answer_dict: 答案字典，用于存储子查询结果
        detail_log: 用于记录详细信息的字典
        
    Returns:
        子查询的最终答案
    """
    # 从DAG节点获取当前深度
    current_depth = planner.global_dag.nodes[subquery].get('depth', 0)
    
    # 为当前子查询初始化详细日志条目
    if subquery not in detail_log:
        detail_log[subquery] = {
            'depth': current_depth,
            'status': 'processing',
            'retrieved_docs': None,
            'answer': None,
            'subqueries': []
        }
    
    # 检查是否达到最大递归深度或图的节点数超过限制
    if current_depth >= RAGConfig.MAX_RECURSION_DEPTH:
        logger.warning(f"子查询达到最大递归深度 {RAGConfig.MAX_RECURSION_DEPTH}，停止切分: {subquery[:50]}...")
        # 即使达到最大深度仍尝试检索
        docs = retriever.search(subquery, top_n=RAGConfig.TOP_N)
        if generator.judge_doc_efficient(subquery, docs):
            logger.warning(f"子查询文档有效性检查通过，生成答案: {subquery[:50]}...")
        else:
            logger.warning(f"子查询文档有效性检查未通过，可能影响答案质量: {subquery[:50]}...")
            docs = [retriever.search(subquery, top_n=RAGConfig.TOP_N+1)[RAGConfig.TOP_N]]  # 增加检索文档数量以提高覆盖率
            if generator.judge_doc_efficient(subquery, docs):
                logger.warning(f"增加检索文档后，子查询文档有效性检查通过: {subquery[:50]}...")
            else:
                logger.warning(f"增加检索文档后，子查询文档有效性检查仍未通过: {subquery[:50]}...")
                docs = pre_retriever.search(subquery, top_n=RAGConfig.TOP_N)  # 使用预检索器结果以增加多样性
        # 记录召回的文档
        detail_log[subquery]['retrieved_docs'] = [doc.get('text') or idx for idx, doc in enumerate(docs)]
        prompt = generator.create_subquery_prompt(subquery, docs)
        result = generator.generate(prompt)
        answer_dict[subquery] = result
        detail_log[subquery]['answer'] = result
        detail_log[subquery]['status'] = 'completed'
        planner.global_dag.nodes[subquery]['completed'] = True
        return result
    
    try:
        # 提取当前子查询的实体/关键词
        perception_result = perceptor.extract(subquery)
        entities = perception_result.get('keywords', [])
        detail_log[subquery]['entities'] = entities
        
        # 预处理检索并检查覆盖率
        pre_results = pre_retriever.search(subquery)
        # coverage_rate = pre_retriever.tcr_by_llm_judgment(subquery, pre_results)
        coverage_rate = pre_retriever.tcr_by_entity_coverage(pre_results, entities)
        logger.warning(f"子查询覆盖率检查 (深度 {current_depth}): {'达标' if coverage_rate else '不达标'} - {subquery[:50]}...")
        detail_log[subquery]['coverage_rate'] = coverage_rate
        
        if coverage_rate:
            logger.warning(f"子查询覆盖率达标，当前深度 {current_depth}")
            # 覆盖率达标，直接检索并生成答案
            docs = retriever.search(subquery, top_n=RAGConfig.TOP_N)
            # 记录召回的文档ID或索引
            detail_log[subquery]['retrieved_docs'] = [doc.get('text') or idx for idx, doc in enumerate(docs)]
            prompt = generator.create_subquery_prompt(subquery, docs)
            subquery_answer = generator.generate(prompt)
            # 更新答案和DAG状态
            answer_dict[subquery] = subquery_answer
            detail_log[subquery]['answer'] = subquery_answer
            detail_log[subquery]['status'] = 'completed'
            planner.global_dag.nodes[subquery]['completed'] = True
            return subquery_answer
        else:
            # 覆盖率不达标，检查深度后决定是否切分
            logger.warning(f"子查询覆盖率不达标，当前深度 {current_depth}")
            
            # 生成更细粒度的子查询
            anchors = pre_retriever.count_hit_keywords(pre_results, entities)
            sub_dag = planner.generate_subqueries_with_anchors(subquery, anchors)
            
            # 记录生成的子查询
            subqueries = list(sub_dag.nodes)
            # subqueries.remove(subquery)  # 移除当前查询本身
            detail_log[subquery]['subqueries'] = subqueries
            detail_log[subquery]['status'] = 'split'
            
            # 为新生成的子查询设置深度
            new_depth = current_depth + 1
            for node in sub_dag.nodes:
                if node not in planner.global_dag.nodes:  # 确保不覆盖已有节点
                    sub_dag.nodes[node]['depth'] = new_depth
                    sub_dag.nodes[node]['completed'] = False
            
            detail_log[subquery]['anchors'] = anchors
            detail_log[subquery]['loop'] = not nx.is_directed_acyclic_graph(sub_dag)
            planner.merge_dag(sub_dag, subquery)
            return None
            
    except Exception as e:
        logger.error(f"处理子查询时出错: {str(e)}", exc_info=True)
        detail_log[subquery]['status'] = 'error'
        detail_log[subquery]['error'] = str(e)
        # 出错时尝试直接检索
        docs = retriever.search(subquery, top_n=RAGConfig.TOP_N)
        detail_log[subquery]['retrieved_docs'] = [doc.get('id') or idx for idx, doc in enumerate(docs)]
        prompt = generator.create_subquery_prompt(subquery, docs)
        result = generator.generate(prompt)
        answer_dict[subquery] = result
        detail_log[subquery]['answer'] = result
        planner.global_dag.nodes[subquery]['completed'] = True
        return result


def process_single_query(
    query: str,
    corpus: Any,
    retriever: RAGRetriever,
    pre_retriever: RAGPreretriever,
    perceptor: RAGPerceptor,
    generator: RAGGenerator,
    llm_call_func: callable,
    config: RAGConfig
) -> Tuple[str, Dict[str, Any]]:  # 修改：返回查询结果和详细日志
    """处理单个查询，生成预测结果"""
    try:
        # 初始化模块
        planner = RAGPlanner(llm_call_func=llm_call_func, model_id=config.LLM_MODEL_ID)
        answer_dict: Dict[str, str] = {}
        
        # 新增：初始化详细日志字典
        detail_log: Dict[str, Any] = {
            'original_query': query,
            'subqueries': {}  # 存储所有子查询的详细信息
        }
        
        # 提取实体和关键词
        logger.warning(f"处理查询: {query[:50]}...")
        perception_result = perceptor.extract(query)
        entities = perception_result.get('keywords', [])
        detail_log['entities'] = entities
        
        # 预处理检索
        pre_retrieval_results = pre_retriever.search(query)
        # coverage_rate = pre_retriever.tcr_by_llm_judgment(query, pre_retrieval_results)
        coverage_rate = pre_retriever.tcr_by_entity_coverage(pre_retrieval_results, entities)
        
        logger.warning(f"初始查询覆盖率: {'达标' if coverage_rate else '不达标'}")
        detail_log['initial_coverage_rate'] = coverage_rate
        
        if RAGConfig.DATASET_NAME != 'hotpot' or not generator.judge_doc_efficient(query, pre_retrieval_results[:3]):
        # if not coverage_rate or not generator.judge_doc_efficient(query, retriever.search(query, top_n=3)):
            logger.warning(f"初始覆盖率不达标，生成子查询并递归处理")
            anchors = pre_retriever.count_hit_keywords(pre_retrieval_results, entities)
            planner.add_original_query(query)
            detail_log['anchors'] = anchors
            
            # 为原始查询设置深度为0
            planner.global_dag.nodes[query]['depth'] = 0
            planner.global_dag.nodes[query]['completed'] = False
            
            # 生成子查询DAG并合并
            subquery_dag = planner.generate_subqueries_with_anchors(query, anchors)
            
            # 为第一层子查询设置深度为1
            for node in subquery_dag.nodes:
                if node != query and node not in planner.global_dag.nodes:
                    subquery_dag.nodes[node]['depth'] = 1
                    subquery_dag.nodes[node]['completed'] = False
            
            detail_log['initial_subqueries'] = list(subquery_dag.nodes)
            planner.merge_dag(subquery_dag, query)
            
            # 处理子查询（递归）
            next_subqueries = planner.select_next_subqueries(retriever, query, answer_dict, corpus)
            print("-------------------next_subqueries-------------------")
            print(next_subqueries)
            
            while next_subqueries:
                for subquery in next_subqueries:
                    logger.warning(f"开始处理子查询: {subquery[:50]}...")
                    
                    # 递归处理子查询，传入详细日志字典
                    subquery_answer = process_subquery_recursive(
                        subquery=subquery,
                        corpus=corpus,
                        perceptor=perceptor,
                        planner=planner,
                        generator=generator,
                        retriever=retriever,
                        pre_retriever=pre_retriever,
                        answer_dict=answer_dict,
                        detail_log=detail_log['subqueries']  # 传入子查询详细日志
                    )
                
                next_subqueries = planner.select_next_subqueries(retriever, query, answer_dict, corpus)
            
            # 生成最终答案
            final_prompt = generator.create_final_prompt(query, answer_dict)
            pred = generator.generate(final_prompt)
            detail_log['dag_nodes'] = dict(planner.global_dag.nodes.items())
            detail_log['dag_edges'] = list(planner.global_dag.edges())
            detail_log['loop'] = not nx.is_directed_acyclic_graph(planner.global_dag)
            detail_log['final_pred'] = pred
        else:
            # 覆盖率达标，直接检索生成答案
            logger.warning(f"覆盖率达标，直接检索生成答案")
            docs = retriever.search(query, top_n=3)
            detail_log['retrieved_docs'] = [doc.get('text') or idx for idx, doc in enumerate(docs)]
            prompt = generator.create_subquery_prompt(query, docs)
            pred = generator.generate(prompt)
            detail_log['final_pred'] = pred
            
        logger.warning(f"查询处理完成，生成预测结果")
        return pred, detail_log  # 修改：返回预测结果和详细日志
        
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
        # 出错时也返回错误信息
        error_log = {
            'original_query': query,
            'status': 'error',
            'error': str(e)
        }
        pred = "查询处理异常"
        raise (pred, error_log)


def _process_one_task(i, query, answer, corpus, config, shared_modules=None):
    """
    处理单个查询（包含重试逻辑）。
    如果 shared_modules 提供 {'perceptor': ..., 'generator': ...} 则尝试复用（调用时需外部加锁）。
    返回 (i, result_dict, query_detail_dict)
    """
    pred = "处理失败"
    query_detail = None

    # 函数内部创建实例（更稳健），或使用 shared modules（如果提供）
    use_shared = bool(shared_modules)
    # A helper to obtain perceptor/generator (either shared or new)
    def make_modules():
        if use_shared:
            return shared_modules['perceptor'], shared_modules['generator']
        # 每个任务单独创建实例，避免并发问题（可能较慢）
        perceptor = RAGPerceptor(model_id=config.LLM_MODEL_ID)
        generator = RAGGenerator(llm_call_func=get_llm_answer, model_id=config.LLM_MODEL_ID)
        return perceptor, generator

    for attempt in range(max(1, getattr(config, "MAX_RETRY", 3))):
        try:
            if use_shared:
                # shared modules 由外部在调用时通过 lock 做并发保护
                perceptor, generator, retriever, preretriever, shared_lock = shared_modules['perceptor'], shared_modules['generator'], shared_modules['retriever'], shared_modules['preretriever'], shared_modules.get('lock')
                # 如果提供 lock，则在调用核心函数时上锁（假设模块不是线程安全）
                if shared_lock:
                    with shared_lock:
                        pred, query_detail = process_single_query(
                            query=query,
                            corpus=corpus,
                            retriever=retriever,
                            pre_retriever=preretriever,
                            perceptor=perceptor,
                            generator=generator,
                            llm_call_func=get_llm_answer,
                            config=config
                        )
                else:
                    pred, query_detail = process_single_query(
                        query=query,
                        corpus=corpus,
                        retriever=retriever,
                        pre_retriever=preretriever,
                        perceptor=perceptor,
                        generator=generator,
                        llm_call_func=get_llm_answer,
                        config=config
                    )
            else:
                perceptor, generator = make_modules()
                pred, query_detail = process_single_query(
                    query=query,
                    corpus=corpus,
                    retriever=retriever,
                    pre_retriever=preretriever,
                    perceptor=perceptor,
                    generator=generator,
                    llm_call_func=get_llm_answer,
                    config=config
                )
            break  # 成功则跳出重试
        except Exception as e:
            # 记录异常并在最后一次尝试失败时返回失败详情
            err_text = "".join(traceback.format_exception_only(type(e), e)).strip()
            logger.warning(f"处理第 {i+1} 条查询失败（尝试 {attempt+1}/{config.MAX_RETRY}）: {err_text}")
            if attempt == config.MAX_RETRY - 1:
                logger.error(f"第 {i+1} 条查询达到最大重试次数")
                query_detail = {
                    'original_query': query,
                    'status': 'failed_after_retries',
                    'error': err_text
                }

    result = {
        'query_id': i,
        'query': query,
        'answer': answer,
        'pred': pred
    }
    if query_detail is None:
        # 保证有基本的 detail 结构
        query_detail = {'original_query': query, 'status': 'success' if pred != "处理失败" else 'unknown'}
    query_detail['query_id'] = i
    query_detail['answer'] = answer

    return i, result, query_detail

def main():
    """主函数，支持并发或串行执行"""
    try:
        # 加载配置和数据
        config = RAGConfig()
        queries, answers, corpus = load_test_data(config.TEST_DATA_PATH)

        # 输出目录
        config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.DETAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # 并发控制参数（从 config 读取或使用默认）
        do_concurrent = getattr(config, "CONCURRENT", True)
        max_workers = int(getattr(config, "MAX_WORKERS", 10))
        share_modules = bool(getattr(config, "SHARE_MODULES", True))

        # 如果选择共享模块，则在主线程初始化并放入 shared_modules
        shared_modules = None
        if do_concurrent and share_modules:
            # 共享实例 — 注意：仅在你确认这些类的初始化可被多个线程共享或你将通过锁保护调用时使用
            shared_lock = threading.Lock() if getattr(config, "SHARE_MODULES_USE_LOCK", True) else None
            perceptor_shared = RAGPerceptor(model_id=config.LLM_MODEL_ID)
            generator_shared = RAGGenerator(llm_call_func=get_llm_answer, model_id=config.LLM_MODEL_ID)
            pre_retriever_shared = RAGPreretriever(corpus)
            # 替换embedding_dir中的硬编码hotpot为数据集变量
            retriever_shared = RAGRetriever(
                corpus, 
                vectorizer_type="bailian", 
                bailian_api_key="your_api_key", 
                embedding_dir=str(config.EMBEDDING_CORPUS_DIR)
            )
            shared_modules = {'perceptor': perceptor_shared, 'generator': generator_shared, 'preretriever': pre_retriever_shared, 'retriever': retriever_shared, 'lock': shared_lock}

        # 文件写入锁（并发时保护写入）
        write_lock = threading.Lock()

        # 打开输出文件（写入所有结果的 jsonl）
        with open(config.OUTPUT_PATH, "w", encoding='utf-8') as outfile:
            if not do_concurrent:
                # 串行执行：与原始逻辑一致
                for i, (query, answer) in tqdm(
                    enumerate(zip(queries, answers)),
                    total=len(queries),
                    desc="处理查询（串行）"
                ):
                    _, result, query_detail = _process_one_task(i, query, answer, corpus, config, shared_modules=None)
                    # 写结果（线程安全操作不过此时是串行）
                    json.dump(result, outfile, ensure_ascii=False)
                    outfile.write('\n')

                    # 写详细日志
                    detail_path = config.DETAIL_LOG_DIR / f"query_detail_{i}.log"
                    with open(detail_path, "w", encoding='utf-8') as detail_file:
                        json.dump(query_detail, detail_file, ensure_ascii=False, indent=2)
                        detail_file.write('\n')

            else:
                # 并发执行路径
                total = len(queries)
                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for i, (query, answer) in enumerate(zip(queries, answers)):
                        # 提交任务，传入 shared_modules（可能为 None）
                        futures.append(ex.submit(_process_one_task, i, query, answer, corpus, config, shared_modules))

                    # 使用 as_completed 逐个获取完成的 future，并更新进度条
                    for fut in tqdm(as_completed(futures), total=total, desc="处理查询（并发）"):
                        try:
                            i, result, query_detail = fut.result()
                        except Exception as e:
                            # 任务本身异常（理论上 _process_one_task 内已捕获），但仍做兜底
                            err_text = "".join(traceback.format_exception_only(type(e), e)).strip()
                            logger.error(f"任务异常: {err_text}")
                            continue

                        # 将结果写入输出文件（加锁）
                        with write_lock:
                            json.dump(result, outfile, ensure_ascii=False)
                            outfile.write('\n')

                        # 写详细日志（每个任务单独文件，不需全局锁，但仍使用锁以防 filesystem race）
                        detail_path = config.DETAIL_LOG_DIR / f"query_detail_{i}.log"
                        with write_lock:
                            with open(detail_path, "w", encoding='utf-8') as detail_file:
                                json.dump(query_detail, detail_file, ensure_ascii=False, indent=2)
                                detail_file.write('\n')

        logger.warning("所有查询处理完成，结果已保存")
        logger.warning(f"详细日志已保存至: {config.DETAIL_LOG_DIR}")

    except Exception as e:
        logger.critical(f"系统运行失败: {str(e)}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()