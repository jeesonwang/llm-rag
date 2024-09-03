#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: split.py
Time: 20240902
"""

import os
from typing import List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

from common.conf import EMBEDDING_MODEL_NAME

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    "。",
    "."
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


if __name__ == "__main__":
    folder_path = "/raid/shnu/wjs/agent/data"

    RAW_KNOWLEDGE_BASE = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            document_id = file_name.split(".")[0]
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                RAW_KNOWLEDGE_BASE.append(LangchainDocument(page_content=text, metadata={"source": document_id}))

    print("Total documents:", len(RAW_KNOWLEDGE_BASE))

    docs_processed = split_documents(
    256,
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    print("Total documents after splitting:", len(docs_processed))
    