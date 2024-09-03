#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: index.py
Time: 20240902
"""

import os
from typing import Dict, List, Tuple
import time

import pandas as pd
from tqdm import tqdm
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from common.log import logger
from utlis.preprocess import preprocess_examples


class Index(object):
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.appname = config['appname']
        self.EMBEDDING_MODEL_NAME = config['embedding_model']
        self.embedding_model = HuggingFaceEmbeddings(
                            model_name=self.EMBEDDING_MODEL_NAME,
                            multi_process=False,
                            model_kwargs={"device": "cuda"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
        self.preprocessor_func = preprocess_examples
        
        self.index = None
        self.load_or_build_index()
        
    def _read_and_preprocess_examples(self) -> List[Document]:
        """
        读取训练集文件并预处理

        Returns:
            Dict: 预处理后的训练集，包含ids, documents, embeddings, metadatas
        """
        processed_examples = self.preprocessor_func(self.config['documents_path'])
        logger.info(f'[{self.appname}] preprocessed {len(processed_examples)} examples from {self.config["documents_path"]}.')
        return processed_examples
    
    def load_or_build_index(self) -> None:
        """
        读取已有索引或构建索引
        This function checks if the index already exists in the specified database. If it does, it loads the index from the database. If it doesn't, it builds the index.
        """
        NotImplemented
        
    
    def _build_index(self) -> None:
        """
        构建索引
        """
        NotImplemented
    
    def query(self, text: str, topk: int = 10) -> Dict:
        """
        查询最相似topk

        Args:
            text (str): 查询文本
            topk (int, optional): 查询结果topk. Defaults to 10.

        Returns:
            Dict: 查询结果
        """
        NotImplemented

    def count(self) -> int:
        """
        A function that counts the number of elements in the index.

        Returns:
            int: The count of elements in the index.
        """
        NotImplemented
    

class ChromadbIndex(Index):
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        

    def load_or_build_index(self) -> None:
        """
        读取已有索引或构建索引
        This function checks if the index already exists in the specified database. If it does, it loads the index from the database. If it doesn't, it builds the index.
        """
        

        db = self.config['database']
        self.index = Chroma(persist_directory=db["address"], collection_name=self.appname, embedding_function=self.embedding_model)
        if self.count() > 0:
            logger.info(f'[{self.appname}] load index from {db["address"]}, count: {self.count()}.')
        else:
            self._build_index()
        

    def _build_index(self) -> None:
        """
        构建索引
        """
        db = self.config['database']
        processed_docs = self._read_and_preprocess_examples()
        logger.info(f'[{self.appname}] build index ...')
        self.index.add_documents(processed_docs)
        self.index.persist()
        logger.info(f'[{self.appname}] build index done.')

    
    def query(self, text: str, topk: int = 10) -> List[Tuple[Document, float]]:
        """
        查询最相似topk

        Args:
            text (str): 查询文本
            keywords (List[str], optional): 通过关键词来筛选topk, keywords中的元素越靠前重要性越高
            topk (int, optional): 查询结果topk. Defaults to 10.

        Returns:
            Dict: 查询结果
        """
        if self.index is None:
            raise Exception(f'[{self.appname}] build index first.')
        results = self.index.similarity_search_with_score(text, k=topk)
        st = time.time()
        
        return results

    def count(self) -> int:
        """
        A function that counts the number of elements in the index.

        Returns:
            int: The count of elements in the index.
        """
        if self.index is None:
            raise Exception(f'[{self.appname}] build index first.')
        return self.index.__len__()

    def to_embedding(self, query: str | list) -> List[List[float]]:
        """
        将文本转化为向量
        """
        return self.embedding_model.embed_documents(query)