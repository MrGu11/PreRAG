import sys
from pathlib import Path
import json
# 获取上级目录路径并添加到sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from search.bm25 import BM25TextMatcher
from search.embedding import VectorRetrievalSystem
from llm.llm import get_llm_answer
from typing import List, Dict, Any, Optional
from textwrap import dedent

class RAGPreretriever:
    def __init__(self, 
                 corpus: List[Dict[str, str]], 
                 BM25matcher = BM25TextMatcher,
                 llm_call_func: Optional[callable] = get_llm_answer):
        """
        Initialize RAG Pre-retriever
        
        Args:
            corpus: Document corpus, each document should be a dict with 'text' key
            BM25matcher: BM25 text matcher class
            llm_call_func: LLM calling function for LLM-based TCR judgment (should accept prompt and model_id)
        """
        # self.matcher = BM25matcher(corpus)
        self.matcher = VectorRetrievalSystem(
            corpus=corpus, 
            vectorizer_type="bailian", 
            bailian_api_key="sk-a1ed3a6082aa4db3b9a5e390805a0ddd", 
            embedding_dir='/data0/home/qwen/thgu/PreRAG/data/embedding/hotpot_corpus_embeddings'
        )
        self.llm_call_func = llm_call_func
        self.corpus = corpus  # Store corpus for later use

    def search(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Calculate text matching scores between query and corpus, return top_n documents
        
        Args:
            query: Search query
            top_n: Number of top documents to return
        
        Returns:
            Sorted list of documents with text and matching scores
        """
        return self.matcher.search(query, top_n)

    # Original entity coverage-based TCR method (retained)
    def tcr_by_entity_coverage(self, results: List[Dict[str, str]], entities: List[str]) -> bool:
        """
        Judge if documents can answer the question through entity coverage: check if any document covers all entities
        
        Args:
            results: List of retrieved documents
            entities: List of entities in the query
        
        Returns:
            Whether there exists a document covering all entities
        """
        if not entities:  # Return True if entity list is empty
            return True
        return any(
            all(entity.lower() in doc['text'].lower() for entity in entities) 
            for doc in results
        )

    # New: LLM-based TCR judgment method
    def tcr_by_llm_judgment(self, 
                           query: str, 
                           results: List[Dict[str, str]], 
                           model_id: str = "qwen3-32b",
                           max_docs: int = 3) -> bool:
        """
        Use LLM to judge if retrieved documents can answer the question
        
        Args:
            query: Original question
            results: List of retrieved documents
            model_id: LLM model ID
            max_docs: Maximum number of input documents (to avoid excessive context length)
        
        Returns:
            LLM's judgment on whether documents can answer the question (True/False)
        """
        if not self.llm_call_func:
            raise ValueError("LLM call function (llm_call_func) not provided, cannot perform LLM-based TCR")
        
        if not results:  # Return False if document list is empty
            return False
        
        # Take top max_docs documents
        input_docs = results[:max_docs]
        
        # Build prompt
        prompt = self._build_tcr_llm_prompt(query, input_docs)
        
        # Call LLM
        try:
            response = self.llm_call_func(prompt, model_id).strip().lower()
            # Judge response (support multiple possible affirmative/negative expressions)
            return any(keyword in response for keyword in ["yes", "able", "can", "true", "sufficient"])
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            return False  # Default to False on call failure

    def _build_tcr_llm_prompt(self, query: str, docs: List[Dict[str, str]]) -> str:
        """Build prompt for LLM-based TCR judgment"""
        # Format document content
        formatted_docs = "\n\n".join([
            f"Document {i+1} content:\n{doc['text'][:1000]}"  # Truncate each doc to first 1000 chars
            for i, doc in enumerate(docs)
        ])
        
        return dedent(f"""
        Task: Determine if the provided documents contain sufficient information to answer the following question.

        Question: {query}

        Reference Documents:
        {formatted_docs}

        Judgment Rules:
        1. If you believe the information in these documents is sufficient to answer the question (no additional information needed), answer "YES"
        2. If you believe the information in these documents is insufficient to answer the question (more information needed), answer "NO"
        3. Only output "YES" or "NO" with no additional explanation.

        Your answer:
        """).strip()

    # def count_hit_keywords(self, results: List[Dict[str, str]], entities: List[str]) -> List[List[str]]:
    #     """
    #     Count keyword sets from entities hit in each document, select disjoint keyword sets as retrieval anchors
        
    #     Args:
    #         results: List of retrieved documents
    #         entities: List of entities in the query
        
    #     Returns:
    #         List of disjoint keyword sets (retrieval anchors)
    #     """
    #     keyword_sets = []
    #     for result in results:
    #         doc_keywords = [
    #             entity for entity in entities 
    #             # if all(word in result['text'].lower() for word in entity.lower().split())
    #             if entity.lower() in result['text'].lower()
    #         ]
    #         if doc_keywords:  # Keep only non-empty keyword sets
    #             # keyword_sets.append(list(set(doc_keywords)))
    #             keyword_sets.append(doc_keywords)
        
    #     # 对每个关键词集合按大小降序排序，能保证选出的锚点覆盖更多关键词
    #     keyword_sets.sort(key=lambda x: -len(x))  # Sort by descending size
    #     # Filter disjoint keyword sets as anchors
    #     anchors = []
    #     for keywords in keyword_sets:
    #         # Check if current keyword set is disjoint from all selected anchors
    #         if all(set(keywords).isdisjoint(anchor) for anchor in anchors):
    #             anchors.append(keywords)
        
    #     return anchors

    def llm_entity_hit(self, entity: str, doc_text: str) -> bool:
        """
        Use LLM to judge whether the entity semantically appears in the document
        """
        prompt = f"""
You are a semantic matching assistant.

Entity:
"{entity}"

Document:
\"\"\"
{doc_text}
\"\"\"

Question:
Does the document explicitly or implicitly mention the entity
(including synonyms, aliases, coreference, or equivalent expressions)?

Answer strictly with one word: Yes or No.
"""
        answer = get_llm_answer(prompt, model_id='qwen3-14b').strip().lower()
        return answer.startswith("yes")

    def count_hit_keywords(
        self,
        results: List[Dict[str, str]],
        entities: List[str]
    ) -> List[List[str]]:
        """
        Count entity sets semantically hit in each document using LLM,
        then select disjoint keyword sets as retrieval anchors
        """
        keyword_sets = []

        for result in results:
            doc_text = result["text"]

            doc_keywords = [
                entity
                for entity in entities
                if self.llm_entity_hit(entity, doc_text)
            ]

            # doc_keywords = [
            #     entity
            #     for entity in entities
            #     if entity.lower() in doc_text.lower()
            # ]

            if doc_keywords:
                keyword_sets.append(doc_keywords)

        # 按覆盖 entity 数量排序（优先选信息量大的 doc）
        keyword_sets.sort(key=lambda x: -len(x))

        # 选取彼此不相交的 anchor
        anchors = []
        for keywords in keyword_sets:
            if all(set(keywords).isdisjoint(anchor) for anchor in anchors):
                anchors.append(keywords)

        return anchors
    
if __name__ == "__main__":
    with open("/data0/thgu/query-decomposition/RAG/data/musique_data/musique_ans_v1.0_train.jsonl", "r", encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i > 3:
                break
            data = json.loads(line.strip())
    sample_corpus = [{"id": paragraph["idx"], "text": paragraph["paragraph_text"]} for paragraph in data["paragraphs"]]

    # # 示例语料库
    # sample_corpus = [
    #     {"id": 1, "text": "The Scarecrow Oz character Illustration by W.W. Denslow from The Wonderful Wizard of Oz First appearance The Wonderful Wizard of Oz (1900) Created by L. Frank Baum Portrayed by Ray Bolger Voiced by Paul Scheer (Once Upon a Time) Information Aliases Socrates Strawman Chang Wang Woe Fiyero Tigelaar Species Scarecrow Gender Male Occupation Ruler of Oz Tin Woodman's treasurer Corn farmer Title His Majesty the Scarecrow Royal Treasurer Emperor of the Silver Islands Spouse (s) Tsing Tsing (in his former incarnation) Children 3 sons 15 grandsons (from his former incarnation)"},
    #     {"id": 2, "text": "The player assumes control of a character referred to as the ``Sole Survivor '', who emerges from a long - term cryogenic stasis in Vault 111, an underground nuclear fallout shelter. After witnessing the murder of their spouse and kidnapping of their son, the Sole Survivor ventures out into the Commonwealth to search for their missing child. The player explores the game's dilapidated world, complete various quests, help out factions, and acquire experience points to level up and increase the abilities of their character. New features to the series include the ability to develop and manage settlements, and an extensive crafting system where materials scavenged from the environment can be used to craft drugs and explosives, upgrade weapons and armor, and construct, furnish and improve settlements. Fallout 4 also marks the first game in the series to feature full voice acting for the protagonist."},
    #     {"id": 3, "text": "Dave Fennoy Fennoy in Phoenix, Arizona David Henderson Fennoy (1952 - 01 - 20) January 20, 1952 (age 65) Silver Spring, Maryland, U.S. Nationality American Occupation Voice actor Years active 1990 -- present Known for The Walking Dead as Lee Everett Minecraft: Story Mode as Gabriel the Warrior Spouse (s) Monique Fennoy Children Michelle Fennoy"},
    #     {"id": 4, "text": "William Everett ``Bud ''Luckey (July 28, 1934 -- February 24, 2018) was an American animator and actor. He best known for his work at Pixar, where he worked as a character designer on a number of films, including Toy Story, Boundin ', Toy Story 2, A Bug's Life, Monsters, Inc., Finding Nemo, Cars and Ratatouille. Luckey was also known as the voice of Rick Dicker in The Incredibles, Chuckles the Clown in Toy Story 3 and as Eeyore in the 2011 Winnie the Pooh film."},
    #     {"id": 5, "text": "Pointless Nostalgic is Jamie Cullum's second album but his first major release on a record label. It was released in 2002 through Candid Records. It was recorded at Clowns Pocket Recording Studio, Bexley, Kent by Derek Nash who also co produced the CD."}
    # ]
    
    # 初始化文本匹配器
    pre_retriever = RAGPreretriever(sample_corpus)
    
    # # 执行查询
    # query = "Who is the spouse of krusty the clown's voice actor?"
    query = data["question"]
    print(query)

    # 建立Perceptor实例
    perceptor = RAGPerceptor()
    
    # 提取query关键词
    item = perceptor.extract(query)
    entities = item['keywords']
    print(entities)

    results = pre_retriever.search(query)
    
    print(pre_retriever.tcr(results, entities))

    anchors = pre_retriever.count_hit_keywords(results, entities)

    print(anchors)
        

