#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: rge.py
Time: 20240902
"""

from typing import Dict, Tuple, Union, List

from models import ChromadbIndex, ReLLM

prompt_template = """
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
"""

class RAGController:
    def __init__(self, config: Dict):
        self.index = ChromadbIndex(config)
        # self.rerank = Rerank()
        self.llm = ReLLM(config)
    
    def __call__(self, query: str, topk: int = 3) -> Tuple[str, List[Dict]]:
        results = self.index.query(text=query, topk=topk)
        
        context = ""
        for result, score in results:
            context += f"Document ID: {result.metadata['source']}, Paragraph ID: {result.metadata['paragraph_id']}: {result.page_content}\n"
        prompt_ = prompt_template.format(context=context, question=query)

        response = self.llm.generate(prompt_)

        return response

