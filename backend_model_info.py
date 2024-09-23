"""
        CAPSTONE TEAM:
    This program will inevitably work with more models than the gpt family models below. When adding a model, make sure everything here looks right.
    
    CHANGES:
    
    Moved model class definition dictionary here.
    Put all variables into a container class.
    
    TODO:
    
    load_in_8bit, used in the generalized models below, will be deprecated as an argument in the future. they should be passed as a BitsAndBytesConfig object.
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
        'gptj': 1
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
