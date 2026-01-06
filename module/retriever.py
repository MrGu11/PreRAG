import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
# 获取上级目录路径并添加到sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from search.embedding import VectorRetrievalSystem as Embeddingmatcher


class RAGRetriever:
    def __init__(self, corpus: List[Dict], model_path: str = "text-embedding-v4", vectorizer_type: str = "bailian", bailian_api_key: Optional[str] = None, embedding_dir: Optional[str] = None):
        self.matcher = Embeddingmatcher(corpus=corpus, model_name=model_path, embedding_dir=embedding_dir, vectorizer_type=vectorizer_type, bailian_api_key=bailian_api_key)

    def search(self, query: str, top_n: int = 1):
        """
        计算query和corpus的相似度分数，并返回前top_n篇
        """
        return self.matcher.search(query, top_n)
    
if __name__ == "__main__":
    with open("/data0/thgu/query-decomposition/RAG/data/musique_data/musique_ans_v1.0_train.jsonl", "r", encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i > 3:
                break
            data = json.loads(line.strip())
    sample_corpus = [{"id": paragraph["idx"], "text": paragraph["paragraph_text"]} for paragraph in data["paragraphs"]]

    
    # 初始化文本匹配器
    retriever = RAGRetriever(sample_corpus)
    
    # # 执行查询
    # query = "Who is the spouse of krusty the clown's voice actor?"
    query = data["question"]
    print(query)

    results = retriever.search(query)

    print(results)
