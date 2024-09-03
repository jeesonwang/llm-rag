#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: llm.py
Time: 20240902
"""

from typing import List, Dict, Any, Optional
import time
import os

from openai import OpenAI
from common.conf import API_KEY, BASE_URL
from common.log import logger

INSTRUCTION = """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Your answer must include Document ID and Paragraph ID of the source document when relevant. Your answer format:
$Answer (Document ID: $Document ID, Paragraph ID: $Paragraph ID).
If the answer cannot be deduced from the context, do not give an answer."""

class LLM(object):
    
    def __init__(self):
        NotImplemented
    
    def generate(self, prompt: str) -> List:
        NotImplemented


class ReLLM(LLM):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        )
        self._config = config
        llm_config = config['llm']
        self.model_name = llm_config['model']
        self.generate_params = llm_config['generate_params']

    def generate(self, prompt: str) -> Dict:
        st = time.time()
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        generate_params = self.generate_params or {}
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generate_params
                )
        except Exception as e:
            logger.error(f"request [{self.model_name}] failed with error {e}")
            raise e
        cost = (time.time()-st) * 1000
        logger.info(f"request [{self.model_name}] response in {cost:.02f} ms ")
        return completion.model_dump()['choices'][0]['message']

