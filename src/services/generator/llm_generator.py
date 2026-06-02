from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from functools import lru_cache
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class SafeLLMGenerator:
    def __init__(self):
        logger.info(f"Loading model: {settings.LLM_MODEL}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
            # local_files_only=True
        ).to("cpu")
        
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():

            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return self._extract_answer(text)