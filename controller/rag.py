#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: rge.py
Time: 20240902
"""

from typing import Dict, Tuple, Union, List

from langchain_core.documents import Document

from models import ChromadbIndex, ReLLM, RAGRerank

prompt_template = """
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
"""

class RAGController:
    """RAG Controller
    
    Example:
        .. code-block:: python
        
            config = {
            "appname": "rag",
            "documents_path": "./data",
            "database": {
                "address": "./data/db",
                "distance_strategy": "cosine"
                },
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "llm": {
                "model": "qwen-max",
                "generate_params": {
                    "max_tokens": 1024,
                    "temperature": 0.01,
                    "top_p": 0.7
                }
            }
            "rerank": {
                "model": "BAAI/bge-reranker-base"
                }
            }
        
            controller = RAGController(config)
    """

    rerank = None
    index = None
    llm = None
    def __init__(self, config: Dict):
        self.index = ChromadbIndex(config)
        if config.get("rerank", False):
            self.rerank = RAGRerank(config)
        self.llm = ReLLM(config)
    
    def __call__(self, query: str, topk: int = 3) -> Tuple[str, List[Dict]]:

        results: List[Tuple[Document, float]] = self.index.query(text=query, topk=10)
        if self.rerank:
            pairs = [(query, result.page_content) for result, _ in results]
            scores = self.rerank.score_pairs(pairs)
            for i, (result, _) in enumerate(results):
                results[i] = (result, scores[i])
            results = sorted(results, key=lambda x: x[1], reverse=True)[:topk]
        context = ""
        for result, score in results:
            context += f"Document ID: {result.metadata['source']}, Paragraph ID: {result.metadata['paragraph_id']}: {result.page_content}\n"
        prompt_ = prompt_template.format(context=context, question=query)
        response = self.llm.generate(prompt_)

        return response

