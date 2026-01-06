        prompt = dedent(f"""
        You are a professional information retrieval assistant. Please process user queries according to the following requirements:

        [Original query]
        {query}

        [Retrieval anchor](The retrieval anchor represents which keyword sets can be obtained through a single article)
        {retrieval_anchors}

        [Task requirements]
        1. Use the search anchor as the splitting boundary to split the original query into multiple subqueries (the search anchor can contain misleading keywords).
        2. The subqueries must be logically clear and logically coherent between subqueries with minimal redundancy to ensure that the combination can accurately restore the main query intent.
        3. Each subquery must point to an answer with "clear conclusions/data/facts". Broad questions without fixed answers such as "How to analyze XX" and "What is the impact of XX" are prohibited.
        4. The dependencies is a list of dependencies, each element is a list [subquery, parent subquery].
            [1] [subquery, parent subquery] indicates that **the completion of the subquery depends on the completion of the parent query**.
            [2] Output query text instead of query number. 
            [2] A query cannot depend on itself.
            [3] Dependency graphs do not allow cycles.
        5. If the original query is a simple factual question (requiring only one answer), no splitting is required and the original query is returned directly with dependencies empty.

        [Example]
        query: "Who is the spouse of the Green performer?"
        subqueries: ["Green >> performer", "[Green >> performer] >> spouse"]
        dependencies: [["[Green >> performer] >> spouse", "Green >> performer"]]

        query: "In what era did The Presbyterian Church in the country whose government disagreed on the language used to describe the torch route experience a large growth in members?"
        subqueries: ["What government disagreed on the language used to describe the torch route?", "In what era did The Presbyterian Church in [What government disagreed on the language used to describe the torch route?] experience a large growth in members?"]
        dependencies:["In what era did The Presbyterian Church in [What government disagreed on the language used to describe the torch route?] experience a large growth in members?", "What government disagreed on the language used to describe the torch route?"]


        【Output format】(json)
        {{"query": original_query, "subqueries": [subquery1, subquery2, ...], "dependencies": dependency_list, "CoT": "Connect the subqueries in series according to their dependencies and compare them with the original query logic."}}
        """