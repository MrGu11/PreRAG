import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
from pathlib import Path
import json
from pdb import set_trace as st


class BaseVectorizer:
    """向量器基类，定义统一接口"""
    def __init__(self):
        self.dimension = None  # 向量维度需在子类中定义
        self.device = None     # 设备（仅本地模型使用）

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量数组，子类必须实现"""
        raise NotImplementedError


class HFTransformersVectorizer(BaseVectorizer):
    """使用Hugging Face Transformers的本地文本向量化器"""
    def __init__(self, model_name: str = "/data0/home/qwen/thgu/model/bge-m3", pooling: str = "cls", normalize: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.normalize = normalize
        
        # 获取向量维度
        self.dimension = self.model.config.hidden_size
        
        # 将模型移至GPU（如果可用）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"初始化本地向量器，模型: {model_name}, 维度: {self.dimension}, 设备: {self.device}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """基于注意力掩码的平均池化"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """将文本转换为向量"""
        if not texts:
            return np.array([])
        
        # 分词并编码
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # 将数据移至设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(** inputs)
        
        # 根据选择的池化策略获取句子向量
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]标记
        elif self.pooling == "mean":
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])  # 平均池化
        elif self.pooling == "mean_sqrt":
            # 加权平均池化（位置平方根倒数权重）
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            position_weights = 1.0 / torch.sqrt(torch.arange(1, token_embeddings.size(1) + 1, dtype=torch.float, device=self.device))
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1)
            masked_weights = position_weights * attention_mask.unsqueeze(-1).float()
            embeddings = torch.sum(token_embeddings * masked_weights, dim=1) / torch.sum(masked_weights, dim=1)
        else:
            raise ValueError(f"不支持的池化策略: {self.pooling}")
        
        # 向量归一化
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 转换为numpy数组
        return embeddings.cpu().numpy()


class BaiLianVectorizer(BaseVectorizer):
    """阿里云百炼（DashScope）文本向量化器"""
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "text-embedding-v4",  # 阿里云百炼支持的embedding模型
                 normalize: bool = True,
                 max_batch_size: int = 32):  # 阿里云API批量处理上限（默认32，可调整）
        super().__init__()
        self.api_key = api_key  # 阿里云百炼仅需单个API Key
        self.model_name = model_name
        self.normalize = normalize  # 是否归一化向量（阿里云部分模型支持）
        self.max_batch_size = max_batch_size  # 分批次处理，避免超API限制
        
        # 阿里云百炼embedding模型维度映射（官方文档确认）
        self.dimension = self._get_model_dimension()
        print(f"初始化阿里云百炼向量器，模型: {model_name}, 维度: {self.dimension}")

    def _get_model_dimension(self) -> int:
        """获取阿里云百炼模型的向量维度（基于官方文档）"""
        model_dim_map = {
            # 通用嵌入模型
            "text-embedding-v4": 1024,       # 阿里云新一代通用嵌入（推荐）
            "text-embedding-v3": 1024,       # 第三代通用嵌入
            # BGE系列模型
            "bge-large-zh": 1024,            # BGE大模型（中文优化）
            "bge-base-zh": 768,              # BGE基础模型（中文优化）
            "bge-small-zh": 384,             # BGE小型模型（中文优化）
            # 行业模型
            "medical-embedding-v1": 768,     # 医疗领域嵌入
            "legal-embedding-v1": 768        # 法律领域嵌入
        }
        if self.model_name in model_dim_map:
            return model_dim_map[self.model_name]
        # 未知模型默认1536（阿里云主流新模型维度）
        print(f"警告：未知模型{self.model_name}，默认向量维度1536（请核对阿里云官方文档）")
        return 1536

    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=1, max=5),  # 指数退避等待（1s→2s→4s）
        retry=retry_if_exception_type((requests.exceptions.RequestException, RuntimeError))
    )
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenAI 原生 Embeddings API（使用官方SDK，无requests依赖）
        - 支持批量分块、顺序保证、完整错误处理
        - 依赖：openai>=1.0.0（新版SDK，旧版v0.x不兼容）
        """
        if not texts:
            return []

        # 1. 初始化OpenAI客户端（支持自定义base_url和超时）
        api_key = getattr(self, "api_key", None)
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key (需设置 self.api_key 或环境变量 OPENAI_API_KEY)")

        # 构建客户端（支持代理地址）
        timeout = getattr(self, "request_timeout", 60)
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
        }
        client_kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client = openai.OpenAI(**client_kwargs)

        # 2. 验证模型与批量限制
        model_name = getattr(self, "model_name", None)
        if not model_name:
            raise RuntimeError("Missing model name (self.model_name)")
        
        # OpenAI官方嵌入模型及单批限制（2024-10最新）
        openai_model_config = {
            "text-embedding-v4": {"max_batch": 10, "dimension": 1024},
            # "text-embedding-3-large": {"max_batch": 1000, "dimension": 3072},
            # "text-embedding-ada-002": {"max_batch": 8191, "dimension": 1536},
            # "text-embedding-3-small-001": {"max_batch": 1000, "dimension": 1536},
            # "text-embedding-3-large-001": {"max_batch": 1000, "dimension": 3072},
        }
        if model_name not in openai_model_config:
            raise ValueError(
                f"不支持的模型: {model_name}\n"
                f"支持列表: {list(openai_model_config.keys())}"
            )
        model_max_batch = openai_model_config[model_name]["max_batch"]

        # 3. 确定分块大小（优先实例配置，不超过模型限制）
        chunk_size = min(getattr(self, "max_batch_size", model_max_batch), model_max_batch)

        # 4. 批量处理文本，按块调用API
        embeddings_by_index = {}  # 按全局索引存储，保证顺序一致
        user_id = getattr(self, "user_id", None)  # 可选：追踪调用者

        for start_idx in range(0, len(texts), chunk_size):
            batch_texts = texts[start_idx:start_idx + chunk_size]
            
            try:
                # 调用OpenAI官方SDK的Embeddings接口
                response = client.embeddings.create(
                    model=model_name,
                    input=batch_texts,
                    encoding_format="float",  # 显式指定返回浮点数组
                    user=user_id  # 可选参数，按需传递
                )
            except openai.APIError as e:
                # 处理API业务错误（模型不存在、额度不足等）
                raise RuntimeError(f"OpenAI API 业务错误: {e.message} (错误码: {e.code})") from e
            except openai.APIConnectionError as e:
                # 处理网络连接错误
                raise RuntimeError(f"OpenAI 连接失败: {str(e)}") from e
            except openai.RateLimitError as e:
                # 处理速率限制（429错误）
                raise RuntimeError(f"API速率限制触发: {e.message}，请降低并发或增加延迟") from e
            except openai.AuthenticationError as e:
                # 处理认证错误
                raise RuntimeError(f"API Key认证失败: {str(e)}") from e
            except Exception as e:
                # 其他未知错误
                raise RuntimeError(f"调用OpenAI API失败: {str(e)}") from e

            # 5. 解析响应，按全局索引存储embedding
            for batch_inner_idx, embedding_obj in enumerate(response.data):
                # 计算当前文本在原始列表中的全局索引
                global_idx = start_idx + batch_inner_idx
                # 提取embedding向量（确保为浮点列表）
                embeddings_by_index[global_idx] = list(embedding_obj.embedding)

                # 验证向量维度（避免模型配置错误）
                if len(embedding_obj.embedding) != openai_model_config[model_name]["dimension"]:
                    raise RuntimeError(
                        f"embedding维度不匹配: 预期 {openai_model_config[model_name]['dimension']}，实际 {len(embedding_obj.embedding)}"
                    )

        # 6. 按原始输入顺序拼接结果
        result = []
        for i in range(len(texts)):
            if i not in embeddings_by_index:
                raise RuntimeError(f"索引 {i} 对应的embedding缺失（可能是API调用丢失）")
            result.append(embeddings_by_index[i])

        return result

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量数组（支持批量处理）"""
        if not texts:
            return np.array([])
        
        # 过滤空文本（避免API报错）
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return np.array([])
        
        all_embeddings = []
        # 分批次处理（避免超过API批量限制）
        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i+self.max_batch_size]
            try:
                batch_embeddings = self._call_embedding_api(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"处理批次[{i//self.max_batch_size + 1}]：{len(batch_texts)}条文本→{len(batch_embeddings)}个向量")
            except Exception as e:
                raise RuntimeError(f"批次[{i//self.max_batch_size + 1}]处理失败: {str(e)}") from e
        
        # 转换为numpy数组（确保float32类型，适配faiss索引）
        return np.array(all_embeddings, dtype=np.float32)


class VectorIndex:
    """向量索引（保持原有功能，兼容两种向量器）"""
    def __init__(self, dimension: int, index_type: str = "flat_l2", use_id_map: bool = False):
        self.dimension = dimension
        self.index_type = index_type
        self.use_id_map = use_id_map
        self.index = self._create_index()
        self.id_to_doc = {}
    
    def _create_index(self):
        if self.use_id_map:
            if self.index_type == "flat_l2":
                base_index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "flat_ip":
                base_index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"索引类型 {self.index_type} 不支持ID映射")
            return faiss.IndexIDMap(base_index)
        else:
            if self.index_type == "flat_l2":
                return faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "flat_ip":
                return faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "hnsw":
                hnsw = faiss.IndexHNSWFlat(self.dimension, 32)
                hnsw.hnsw.efConstruction = 40
                return hnsw
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def add_vectors(self, vectors: np.ndarray, documents: List[Dict], ids: Optional[List[int]] = None):
        if ids is None:
            ids = list(range(self.index.ntotal, self.index.ntotal + len(vectors)))
        
        vectors = vectors.astype(np.float32)
        
        if self.use_id_map:
            self.index.add_with_ids(vectors, np.array(ids))
        else:
            self.index.add(vectors)
        
        for doc_id, doc in zip(ids, documents):
            self.id_to_doc[doc_id] = doc
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[Dict]]:
        query_vector = query_vector.astype(np.float32)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        documents = []
        for idx in indices[0]:
            if idx != -1:
                documents.append(self.id_to_doc.get(idx, {}))
        
        return distances[0], documents


class VectorRetrievalSystem:
    """支持本地模型和百炼API的向量检索系统（新增保存/加载功能）"""
    def __init__(self, 
                 corpus: List[Dict] = None, 
                 text_field: str = "text", 
                 embedding_dir: str = None,
                 id_field: str = "id",
                 # 向量器配置
                 vectorizer_type: str = "local",  # "local" 或 "bailian"
                 model_name: str = "/data0/home/qwen/thgu/RAG/model/bge-m3",
                 # 百炼API参数（仅当vectorizer_type="bailian"时需要）
                 bailian_api_key: Optional[str] = None,
                 index_type: str = "flat_l2"):
        # st()
        """初始化检索系统（保持原有参数不变）"""
        self.text_field = text_field
        self.id_field = id_field
        self.vectorizer_type = vectorizer_type
        self.model_name = model_name
        self.bailian_api_key = bailian_api_key  # 保存API密钥（加载时复用）
        self.index_type = index_type
        self.embedding_dir = embedding_dir
        
        # 初始化向量器
        self.vectorizer = self._init_vectorizer(
            vectorizer_type=vectorizer_type,
            model_name=model_name,
            api_key=bailian_api_key
        )
        
        # 初始化索引
        self.index = VectorIndex(
            dimension=self.vectorizer.dimension,
            index_type=index_type,
            use_id_map=(id_field is not None)
        )
        
        # 新增：存储embedding向量（便于后续保存）
        self.embeddings: Optional[np.ndarray] = None
        
        if embedding_dir is not None:
            embedding_dir = Path(embedding_dir)
            if not embedding_dir.exists():
                if corpus is None:
                    raise ValueError("未提供语料库，无法构建新索引")
                self.build_index(corpus)
            else:
                self.load(self.embedding_dir)
        else:
            raise ValueError("必须提供 embedding_dir 参数以保存或加载索引")

    def _init_vectorizer(self, vectorizer_type: str, model_name: str, api_key: Optional[str]) -> BaseVectorizer:
        """初始化向量器（本地/百炼API）"""
        if vectorizer_type == "local":
            return HFTransformersVectorizer(model_name=model_name)
        elif vectorizer_type == "bailian":
            if not api_key:
                raise ValueError("使用百炼API时必须提供bailian_api_key")
            return BaiLianVectorizer(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"不支持的向量器类型: {vectorizer_type}")

    def build_index(self, corpus: List[Dict]):
        """构建向量索引（新增：保存embedding向量）"""
        texts = [doc[self.text_field] for doc in corpus]
        self.embeddings = self.vectorizer.vectorize(texts).astype(np.float32)  # 保存向量
        
        ids = [doc.get(self.id_field) for doc in corpus] if self.id_field is not None else None
        self.index.add_vectors(self.embeddings, corpus, ids)
        print(f"索引构建完成，文档数: {len(corpus)}, 向量维度: {self.vectorizer.dimension}")
        self.save(self.embedding_dir, overwrite=True)  # 自动保存最新索引

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """执行搜索（保持原有逻辑不变）"""
        query_vector = self.vectorizer.vectorize([query])[0].astype(np.float32)
        
        # 内积索引需要归一化查询向量
        if isinstance(self.index.index, faiss.IndexFlatIP) and not hasattr(self.vectorizer, 'normalize'):
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        distances, documents = self.index.search(query_vector, k)
        
        results = []
        for distance, doc in zip(distances, documents):
            if doc:
                # 转换距离为相似度分数
                if isinstance(self.index.index, faiss.IndexFlatL2):
                    score = 1 / (1 + distance)  # L2距离→相似度（0-1）
                else:
                    score = distance  # 内积直接作为相似度
                result = doc.copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def save(self, directory: Path, overwrite: bool = False):
        directory = Path(directory)
        if directory.exists():
            if not overwrite:
                raise FileExistsError(f"目录 {directory} 已存在，设置 overwrite=True 以覆盖")
        else:
            directory.mkdir(parents=True, exist_ok=True)

        # 保存 faiss 索引（若 index 是 wrapper，保存 wrapper）
        index_path = directory / "vector_index.faiss"
        faiss.write_index(self.index.index, str(index_path))

        # 保存配置
        config = {
            "text_field": self.text_field,
            "id_field": self.id_field,
            "vectorizer_type": self.vectorizer_type,
            "model_name": self.model_name,
            "bailian_api_key": self.bailian_api_key,
            "index_type": self.index_type
        }
        with open(directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 保存 embeddings（可选）
        if self.embeddings is not None:
            np.save(str(directory / "embeddings.npy"), self.embeddings)

        # 保存 id_to_doc
        with open(directory / "docs.json", "w", encoding="utf-8") as f:
            # 将 key 转为字符串保存（json 限制），加载时再转回 int
            serializable = {str(k): v for k, v in self.index.id_to_doc.items()}
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        print(f"向量索引和配置已保存到 {directory}")

    def load(self, directory: Path):
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"目录 {directory} 不存在")

        # 读取 config
        with open(directory / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        self.text_field = config.get("text_field")
        self.id_field = config.get("id_field")
        self.vectorizer_type = config.get("vectorizer_type")
        self.model_name = config.get("model_name")
        self.bailian_api_key = config.get("bailian_api_key")
        self.index_type = config.get("index_type")

        # 重新初始化 vectorizer（若必要）
        self.vectorizer = self._init_vectorizer(
            vectorizer_type=self.vectorizer_type,
            model_name=self.model_name,
            api_key=self.bailian_api_key
        )

        # 加载 faiss 索引
        loaded_index = faiss.read_index(str(directory / "vector_index.faiss"))
        # 将 loaded_index 赋给 self.index.index（兼容 wrapper）
        if hasattr(self.index, "index"):
            self.index.index = loaded_index
        else:
            self.index = VectorIndex(dimension=self.vectorizer.dimension,
                                     index_type=self.index_type,
                                     use_id_map=(self.id_field is not None))
            self.index.index = loaded_index

        # 加载 embeddings（如果存在）
        emb_path = directory / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(str(emb_path))
        else:
            self.embeddings = None

        # 加载 docs mapping（id->doc）
        docs_path = directory / "docs.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                serializable = json.load(f)
            # keys stored as strings -> convert back to int
            self.index.id_to_doc = {int(k): v for k, v in serializable.items()}
        else:
            self.index.id_to_doc = {}

        print(f"向量索引和配置已从 {directory} 加载完成")



# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 示例语料
    sample_corpus = [
        {"id": 1, "text": "人工智能是计算机科学的一个分支"},
        {"id": 2, "text": "机器学习是人工智能的一个重要领域"},
        {"id": 3, "text": "深度学习是机器学习的子集，基于神经网络"}
    ]
    
    # # 1. 使用本地模型
    # local_retriever = VectorRetrievalSystem(
    #     corpus=sample_corpus,
    #     vectorizer_type="local",
    #     model_name="/data0/home/qwen/thgu/RAG/model/bge-m3",
    #     index_type="flat_ip"
    # )
    # print("本地模型检索结果:")
    # print(local_retriever.search("什么是深度学习", k=2))

    print("百炼API检索结果:")
    print(bailian_retriever.search("什么是深度学习", k=2))