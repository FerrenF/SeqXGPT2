import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

from config_manager import ConfigManager
from backend_model_info import SeqXGPT2_ModelInfoContainer



class SeqXFeatureGenerator:
          
    
   
    config_models = ConfigManager.get_all_keys()
        
    @classmethod
    def access_api(cls, text, api_url, do_generate=False):
        """
        :param text: input text
        :param api_url: api
        :param do_generate: whether generate or not
        :return:
        """
        with httpx.Client(timeout=None) as client:
            post_data = {
                "text": text,
                "do_generate": do_generate,
            }
            prediction = client.post(url=api_url,
                                    data=msgpack.packb(post_data),
                                    timeout=None)
        if prediction.status_code == 200:
            content = msgpack.unpackb(prediction.content)
        else:
            content = None
        return content

    @classmethod
    def get_features(cls, input_file, output_file, line_limit, shuffle):
        
        """
        get [losses, begin_idx_list, ll_tokens_list, label_int, label] based on raw lines
        """      
      
        # line = {'text': '', 'label': ''}
        with open(input_file, 'r') as f:
            lines = [json.loads(line) for line in f]
        if shuffle:
            random.shuffle(lines)
        if line_limit > 0:
            lines = lines[:line_limit]
        print('input file:{}, length:{}'.format(input_file, len(lines)))

        with open(output_file, 'w', encoding='utf-8') as f:
            for data in tqdm(lines):
                line = data['text']
                label = data['label']

                losses = []
                begin_idx_list = []
                ll_tokens_list = []

                label_dict = SeqXGPT2_ModelInfoContainer.en_labels
                label_int = label_dict[label]

                error_flag = False
                
                for model in cls.config_models:
                    
                    opt = ConfigManager.get_model_args(model)
                    api_addr_host = opt['host']
                    api_addr_port = opt['port']
                    model_api_addr = rf'http://{api_addr_host}:{api_addr_port}/inference'
                    try:
                        loss, begin_word_idx, ll_tokens = cls.access_api(line, model_api_addr)
                    except TypeError as e:
                        print(f"return NoneType, probably gpu OOM, discard this sample. api: {model_api_addr} {e}")
                        error_flag = True
                        break
                    losses.append(loss)
                    begin_idx_list.append(begin_word_idx)
                    ll_tokens_list.append(ll_tokens)
                    
                # if oom, discard this sample
                if error_flag:
                    continue

                result = {
                    'losses': losses,
                    'begin_idx_list': begin_idx_list,
                    'll_tokens_list': ll_tokens_list,
                    'label_int': label_int,
                    'label': label,
                    'text': line
                }

                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file")
    parser.add_argument("--output_file", type=str, help="output file")

    parser.add_argument("--line_limit", default=0, action="store_true", help="Stop at number of lines from input file. Default: 0 (unlimited)")
    parser.add_argument("--get_en_features", action="store_true", help="DEPRECATED generate en logits and losses")
    parser.add_argument("--get_en_features_multithreading", action="store_true", help="multithreading generate en logits and losses")
    parser.add_argument("--do_normalize", action="store_true", help="normalize the features")
    parser.add_argument("--shuffle_lines", default=False, action="store_true", help="shuffle the lines in the input file. Default: False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    feature_generator = SeqXFeatureGenerator()
    
    if not ConfigManager.config_exists():
        print("Config file doesn't exist - inference server may not be running. Exiting application...")
        exit(-1)

    if args.get_en_features:
        """
        retrieve english features in a single file 
        python gen_features.py --get_en_features --input_file raw_data/en_alpaca_lines.jsonl --output_file ../features/raw_features/en_alpaca_features.jsonl
        python gen_features.py --get_en_features --input_file raw_data/en_dolly_lines.jsonl --output_file ../features/raw_features/en_dolly_features.jsonl
        """
        feature_generator.get_features(input_file=args.input_file, output_file=args.output_file, line_limit = args.line_limit, shuffle = args.shuffle_lines)


    elif args.get_en_features_multithreading:
        """
        retrieve english features in multiple files, use multithreading for faster speed
        python gen_features.py --get_en_features_multithreading
        """

        en_input_files = []

        en_output_files = []

        threads = []
        for i in range(len(en_input_files)):
            t = threading.Thread(target=feature_generator.get_features, args=(en_input_files[i], en_output_files[i]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    else:
        print("please select an action")
