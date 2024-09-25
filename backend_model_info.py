"""
        CAPSTONE TEAM:
    This program will inevitably work with more models than the gpt family models below. When adding a model, make sure everything here looks right.
    
    CHANGES:
    
    Moved model class definition dictionary here.
    Put all variables into a container class.
    
    TODO:
    
    load_in_8bit, used in the generalized models below, will be deprecated as an argument in the future. they should be passed as a BitsAndBytesConfig object.
    
    
    ['Bloom-7B'  Megatron - GPT-2 Family 
    'Claude-Instant-v1' - CLOSED SOURCE - PAID
    'Claude-v1' - CLOSED SOURCE - PAID
    'Cohere-Command' - CLOSED SOURCE, PAID | ALTERNATIVE C4ai-command-r-plus-4bit https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit
    'Dolphin-2.5-Mixtral-8x7B' | Llama family model https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF
    'Dolphin-Mixtral-8x7B' 
    'Falcon-180B' - UNFEASIBLE | ALTERNATIVE Falcon-40b https://huggingface.co/tiiuae/falcon-40b
    'Flan-T5-Base' - T5 Family https://huggingface.co/google/flan-t5-base 
    'Flan-T5-Large' - https://huggingface.co/google/flan-t5-large 
    'Flan-T5-Small' - https://huggingface.co/google/flan-t5-small
    'Flan-T5-XL' - https://huggingface.co/google/flan-t5-xl
    'Flan-T5-XXL' - UNFEASIBLE
    'GLM-130B' - UNFEASIBLE https://huggingface.co/ianZzzzzz/GLM-130B-quant-int4-4gpu/tree/main/49300
    'GPT-3.5' - CLOSED SOURCE
    'GPT-4' - CLOSED SOURCE
    'GPT-J' - DONE ******
    'GPT-NeoX' - GPT-NEO Family, BETTER FOR CODING https://huggingface.co/docs/transformers/model_doc/gpt_neox
    'Gemini-Pro'
    'Goliath-120B' 
    'Human' 
    'LLaMA-13B' - https://huggingface.co/yahma/llama-13b-hf 
    'LLaMA-2-70B' 
   *** 'LLaMA-2-7B' - OPEN SOURCE https://huggingface.co/meta-llama/Llama-2-7b
    'LLaMA-30B' - https://huggingface.co/huggyllama/llama-30b
    'LLaMA-65B' 
    'LLaMA-7B' - https://huggingface.co/huggyllama/llama-7b
    'LZLV-70B' 
    'Mistral-7B' - https://huggingface.co/mattshumer/mistral-8x7b-chat
    'Mistral-7B-OpenOrca' Mistral Family - https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca
    'Mixtral-8x7B' - USE QUANTIZED https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ
    'MythoMax-L2-13B' 
    'Neural-Chat-7B' 
    'Noromaid-20B'
    'Nous-Capybara-34B' 
    'Nous-Capybara-7B' 
    'Nous-Hermes-LLaMA-2-13B'
    'Nous-Hermes-LLaMA-2-70B' 
    'OPT-1.3B' - GPT3 FAMILY DECODERS https://huggingface.co/facebook/opt-1.3b
    'OPT-125M' 
    'OPT-13B' - https://huggingface.co/facebook/opt-13b
    'OPT-2.7B'
    'OPT-30B' 
    'OPT-350M' 
    'OPT-6.7B' 
    'OpenChat-3.5' 
    'OpenHermes-2-Mistral-7B'
    'OpenHermes-2.5-Mistral-7B' 
    'PaLM-2' 
    'Psyfighter-13B' https://huggingface.co/TheBloke/Psyfighter-13B-GGUF QUANTIZED
    'Psyfighter-2-13B'
    'RWKV-5-World-3B' 
    'StripedHyena-Nous-7B' - https://huggingface.co/togethercomputer/StripedHyena-Nous-7B
    'T0-11B' 
    'T0-3B' 
    'Text-Ada-001'
    'Text-Babbage-001' 
    'Text-Curie-001' 
    'Text-Davinci-001' 
    'Text-Davinci-002'
    'Text-Davinci-003' 
    'Toppy-M-7B' 
    'Unknown' 
    'YI-34B' - UNFEASIBLE | ALTERNATIVE https://huggingface.co/01-ai/Yi-6B
    ]
 
"""
from collections import OrderedDict

from backend_model import GPT2SnifferModel, GPTNeoSnifferModel, GPTJSnifferModel

class SeqXGPT2_ModelInfoContainer:
    # All models that Sniffer will use.
    en_model_names = ['gpt_2', 'gpt_neo', 'gpt_J']

    # feature
    tot_feat_num = 4
    # Warn: when use 'loss_only' or 'feature_only', 
    # you need change the hidden_size in both train.py and the backend_sniffer.py
    train_feat = 'all'
    cur_feat_num = 4
    # checkpoint path, this is corresponding to the `cur_feat_num` and `train_feat`
    en_ckpt_path = ""

    # The labels our classifier expects for each model
    en_labels = {
        'gpt2': 0,
        'gptneo': 1,
        'gptj': 2,
        'llama2': 3,
        'human' : 5
    }
    en_class_num = 5

    base_model = "roberta-base"

 
    MODEL_MAP = OrderedDict([        
        ("gpt2", GPT2SnifferModel),
        ("gptneo", GPTNeoSnifferModel),
        ("gptj", GPTJSnifferModel)
        ])
    
    learning_feature_directory = "learning/features/"
    learning_raw_directory = "learning/raw/"
