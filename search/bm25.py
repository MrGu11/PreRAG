import math
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# # 下载NLTK资源（如果尚未下载）
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# nltk.download('punkt_tab')
    
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

def download_nltk_resource(resource: str) -> None:
    """
    下载NLTK资源（如果尚未下载）
    
    Args:
        resource: 资源路径，如'tokenizers/punkt'
    """
    try:
        nltk.data.find(resource)
    except LookupError:
        # 提取资源名称（最后一部分）进行下载
        resource_name = resource.split('/')[-1]
        nltk.download(resource_name)


# 确保必要的NLTK资源已下载
download_nltk_resource('tokenizers/punkt_tab')
download_nltk_resource('corpora/stopwords')

class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, tokenizer=None):
        """
        初始化BM25模型
        
        Args:
            corpus: 语料库，每个元素是一个文本
            k1: BM25参数，控制词频的影响程度
            b: BM25参数，控制文档长度的影响程度
            tokenizer: 分词器，默认为英文分词器
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenizer = tokenizer or self._default_tokenizer
        
        # 分词后的语料库
        self.tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        
        # 文档平均长度
        self.avgdl = sum(len(doc_tokens) for doc_tokens in self.tokenized_corpus) / len(self.tokenized_corpus)
        
        # 构建倒排索引
        self.inverted_index = self._build_inverted_index()
        
        # 计算IDF值
        self.idf = self._calculate_idf()
        
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        默认英文分词器
        1. 转换为小写
        2. 去除标点符号
        3. 分词
        4. 去除停用词
        5. 词干提取
        """
        # 转换为小写
        text = text.lower()
        
        # 去除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        tokens = word_tokenize(text)
        # print(tokens)        
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # 词干提取
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def _build_inverted_index(self) -> Dict[str, Set[int]]:
        """构建倒排索引"""
        inverted_index = defaultdict(set)
        for doc_id, doc_tokens in enumerate(self.tokenized_corpus):
            for token in set(doc_tokens):  # 每个词在文档中只记录一次
                inverted_index[token].add(doc_id)
        return inverted_index
    
    def _calculate_idf(self) -> Dict[str, float]:
        """计算每个词的IDF值"""
        idf = {}
        num_docs = len(self.corpus)
        for term, doc_ids in self.inverted_index.items():
            # +1平滑处理，避免IDF为无穷大
            idf[term] = math.log((num_docs - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5) + 1)
        return idf
    
    def get_scores(self, query: str) -> List[float]:
        """
        计算查询与所有文档的相似度得分
        
        Args:
            query: 查询文本
            
        Returns:
            得分列表，每个元素对应一个文档的得分
        """
        query_tokens = self.tokenizer(query)
        scores = [0.0] * len(self.corpus)
        
        for term in query_tokens:
            if term not in self.idf:
                continue
                
            # 计算包含该词的文档列表
            doc_ids = self.inverted_index.get(term, set())
            
            # 计算词频因子
            idf_val = self.idf[term]
            
            for doc_id in doc_ids:
                # 获取文档中该词的词频
                tf = self.tokenized_corpus[doc_id].count(term)
                
                # 获取文档长度
                doc_len = len(self.tokenized_corpus[doc_id])
                
                # 计算BM25得分
                score = idf_val * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                
                # 累加到文档总分
                scores[doc_id] += score
                
        return scores
    
    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        获取与查询最相似的n个文档
        
        Args:
            query: 查询文本
            n: 返回的文档数量
            
        Returns:
            文档列表，每个元素是一个元组(文档内容, 得分)
        """
        scores = self.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.corpus[i], scores[i]) for i in top_indices]

class BM25TextMatcher:
    def __init__(self, corpus: List[Dict], text_field: str = "text", id_field: str = "id"):
        """
        端到端BM25文本匹配器
        
        Args:
            corpus: 语料库，每个元素是一个字典，包含文本和可选的ID
            text_field: 文本字段名
            id_field: ID字段名，可选
        """
        self.corpus = corpus
        self.text_field = text_field
        self.id_field = id_field
        
        # 提取文本内容构建BM25模型
        texts = [item[text_field] for item in corpus]
        self.bm25 = BM25(texts)
        
    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        执行搜索并返回匹配结果
        
        Args:
            query: 查询文本
            top_n: 返回结果数量
            
        Returns:
            匹配结果列表，每个元素是一个字典，包含原始文档信息和匹配得分
        """
        # 获取BM25得分
        scores = self.bm25.get_scores(query)
        
        # 排序并获取前n个结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        # 构建结果
        results = []
        for idx in top_indices:
            result = self.corpus[idx].copy()
            result["score"] = scores[idx]
            results.append(result)
            
        return results

# 使用示例
if __name__ == "__main__":
    # 示例语料库
    sample_corpus = [
        {"id": 1, "text": "Natural language processing is an important field in artificial intelligence."},
        {"id": 2, "text": "Deep learning has achieved great success in computer vision and speech recognition."},
        {"id": 3, "text": "BM25 is a commonly used information retrieval algorithm."},
        {"id": 4, "text": "TF-IDF is another classic text representation method."},
        {"id": 5, "text": "Information retrieval systems usually require efficient text matching techniques."}
    ]
    
    # 初始化文本匹配器
    matcher = BM25TextMatcher(sample_corpus, text_field="text", id_field="id")
    
    # 执行查询
    query = "What is the BM25 algorithm?"
    results = matcher.search(query, top_n=3)
    
    # 打印结果
    print(f"Query: {query}")
    print("Matching Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. ID: {result['id']}, Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
        print()    