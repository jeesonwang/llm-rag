#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: reranker.py
Time: 20240914
"""

from abc import ABC
from typing import List, Dict, Any, Tuple
import time
import requests

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from common.log import logger

class Rerank(ABC):
    
    def __init__(self):
        NotImplemented
    
    def rerank_api(self, input: List[str]) -> List:
        NotImplemented

    def rerank(self, input: List[str]) -> List:
        NotImplemented

class RAGRerank(Rerank):
    reranker = None

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._config = config
        self.model_name = config['model']
        self.url = config.get('url')

    @staticmethod
    def sort_by_sim(list, sim, reverse=True):
        result = sorted(zip(list, sim), key=lambda x: x[1], reverse=reverse)
        return [r[0] for r in result]
        
    def rerank_api(self, inputs: List[List[str]], url: str = None) -> List[float]:
        """
        Sends a POST request to the specified URL with the input documents to obtain rerank similarity.

        Args:
            input (List[List[str]]): The input documents to obtain rerank similarity for.
            example: [["上海天气","北京美食"],["上海天气","上海气候"]]

        Returns:
            List[float]: The rerank similarity obtained from the request.
            example:  [
                7.635734073119238e-05,
                0.9981106519699097
            ]
        """
        rerank_url = url or self.url
        st = time.time()
        response = requests.post(
            rerank_url,
            json={
                "model": self.model_name,
                "input": list(inputs),
            }
        )
        if response.status_code != 200:
            raise ValueError(f"request failed with status code {response.status_code}")
        similarity = response.json()["data"]
        cost = (time.time()-st) * 1000
        logger.info(f"request [{self.model_name}] response in {cost:.02f} ms : {similarity}")
        return similarity
    
    def rerank(self, inputs: List[List[str, str]] | List[Tuple[str, str]], model_name: str = None) -> List[float]:
        if self.reranker is None:
            try:
                self.reranker = HuggingFaceCrossEncoder(model_name=model_name)
                logger.info(f"Load model {model_name} success")
            except Exception as e:
                logger.error(f"Load model {model_name} failed")
                raise e
        similarity = self.reranker.score(inputs)
        return similarity
    
    def score_pairs(self, inputs: List[List[str]] | List[Tuple[str, str]], model_name: str = None) -> List[float]:
        model_name = model_name or self.model_name
        if self.url:
            return self.rerank_api(inputs, self.url)
        else:
            return self.rerank(inputs, model_name)
    