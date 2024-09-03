#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: prerpocess.py
Time: 20240902
"""

import os
from tqdm import tqdm
import pandas as pd
import pathlib
from typing import List, Union, Optional

from langchain.docstore.document import Document as LangchainDocument

from .split import split_documents
from common.conf import EMBEDDING_MODEL_NAME

def preprocess_examples(file_path: str) -> List[LangchainDocument]:
    """预处理流程

    Args:
        file_path (str): _description_

    """

    file_path = pathlib.Path(file_path)
    if file_path.suffix == '':
        folder_path = file_path
        RAW_KNOWLEDGE_BASE = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                cur_file_path = os.path.join(folder_path, file_name)
                document_id = file_name.split(".")[0]
                with open(cur_file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=text, metadata={"source": document_id}))
    # TODO: add more file type
    docs_processed = split_documents(
                        256,
                        RAW_KNOWLEDGE_BASE,
                        tokenizer_name=EMBEDDING_MODEL_NAME,
                        )
    paragraph_id = 1
    for doc in docs_processed:
        if doc.metadata['start_index'] == 0:
            paragraph_id = 1
        doc.metadata['paragraph_id'] = paragraph_id
        paragraph_id += 1
    return docs_processed

