import torch
import transformers
import numpy as np

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc, CharLevelTokenizerPPLCalc, SPChatGLMTokenizerPPLCalc
from backend_utils import split_sentence



# mosec and it's dependencies
from io import BytesIO
from typing import List

from mosec import Worker
from mosec.mixin import MsgpackMixin

from transformers import BitsAndBytesConfig

# llama
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

"""
    Major changes: Rewrite all GPT family models into a single inheriting class with parameters specifying differences. Cut down a lot of code.
"""


class SnifferBaseModel(MsgpackMixin, Worker):
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Determines whether the model will use a GPU (cuda) or fallback to CPU.
        self.do_generate = None # A flag indicating whether to perform generation or perplexity calculation.
        self.text = None
        self.base_tokenizer = None # Set up in inheriting class
        self.base_model = None # set up in inheriting class
        self.generate_len = 512 # Generation token limit

    def forward_calc_ppl(self):
        pass

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            tokenized = self.base_tokenizer(self.text, return_tensors="pt").to(
                self.device)
            tokenized = tokenized.input_ids
            gen_tokens = self.base_model.generate(tokenized,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.squeeze()
            result = self.base_tokenizer.decode(gen_tokens.tolist())
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return gen_texts

    def forward(self, data):
        """
        This overrides the forward method from the mosec worker class. It's the first stop for text received on the backend API running our model.
        
        :param data: ['text': str, "do_generate": bool]
        :return:
        """
        self.text = data["text"]
        self.do_generate = data["do_generate"]
        
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()

# Examples:
# gpt2_model = SnifferModel(model_name='gpt2-xl')
# gpt_neo_model = SnifferModel(model_name='EleutherAI/gpt-neo-2.7B', load_in_8bit=True, device_map="auto")
# gpt_j_model = SnifferModel(model_name='EleutherAI/gpt-j-6B', load_in_8bit=True, device_map="auto")


# In the original code, some tokenizers had their special token values mapped to another value - for various reasons. Like llama, having it's pad_token_id being mapped to the eos_token_id.
class specialTokenMap:
    def __init__(self, tokenizer):
        self.tokenMap = {}
        self.tokenizer = None
    def loadNewMap(self):
         for key, value in self.tokenMap.items():
            if hasattr(self.tokenizer, value):
                # Get the attribute value of the tokenizer
                token_value = getattr(self.tokenizer, value)
            else:
                # If the attribute doesn't exist, set token_value to the string value itself
                token_value = value
            # Now, set the tokenizer attribute identified by 'key'
            setattr(self.tokenizer, key, token_value)      

class specialLlamaTokenMap(specialTokenMap):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.tokenMap["pad_token_id"] = "bos_token_id"
        
class SnifferGeneralFamilyModel(SnifferBaseModel):
    def __init__(self, model_name="gpt2", ppl_calculator_class=BBPETokenizerPPLCalc, tokenizer_class=transformers.AutoTokenizer, quantization_config=None,device_map=None,loadSpecialTokenMap=None,optional_params={}):
        super().__init__()  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.model_name = model_name
        self.ppl_calculator_class = ppl_calculator_class
        self.quantization_config = quantization_config
        self.device_map = device_map
        
          # Load the tokenizer and model dynamically based on the provided model name, and use token if possible
        self.base_tokenizer = tokenizer_class.from_pretrained(self.model_name, **optional_params)
        
        if loadSpecialTokenMap is not None:
            mapper = loadSpecialTokenMap(self.base_tokenizer)
            mapper.loadNewMap()
            
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map, quantization_config=self.quantization_config, **optional_params)
        
        # Set padding token ID
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        
        is_quantized = getattr(self.base_model, "is_quantized", False) or getattr(self.base_model, "is_loaded_in_8bit", False)
        if not is_quantized:
            self.base_model.to(self.device)
        
        # Initialize perplexity calculator
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = ppl_calculator_class(byte_encoder, self.base_model, self.base_tokenizer, self.device)     

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
quant_config_4bit_bfloat = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
quant_config_8bit_bfloat = BitsAndBytesConfig(load_in_8bit=True,  bnb_4bit_compute_dtype=torch.bfloat16)
class GPT2SnifferModel(SnifferGeneralFamilyModel):
    def __init__(self):
        super().__init__(model_name="gpt2")  
                
class GPTNeoSnifferModel(SnifferGeneralFamilyModel):
    def __init__(self):
        super().__init__(model_name="EleutherAI/gpt-neo-2.7B", quantization_config=quant_config_4bit_bfloat, device_map="auto")  
        
class GPTJSnifferModel(SnifferGeneralFamilyModel):
    def __init__(self):
        super().__init__(model_name="EleutherAI/gpt-j-6B",quantization_config=quant_config_4bit_bfloat, device_map="auto")  
        
class LlamaSnifferModel(SnifferGeneralFamilyModel):
    hf_token = "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" # llama2 is gated access
    def __init__(self):
        super().__init__(model_name="meta-llama/Llama-2-7b-hf",quantization_config=quant_config_4bit_bfloat, tokenizer_class=LlamaTokenizer, device_map="auto",
                         loadSpecialTokenMap=specialLlamaTokenMap, ppl_calculator_class=SPLlamaTokenizerPPLCalc, optional_params={'token':LlamaSnifferModel.hf_token})  
        