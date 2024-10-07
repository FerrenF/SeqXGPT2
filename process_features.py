import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import itertools

class SeqXFeatureProcessor:

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", type=str, help="input file")
        parser.add_argument("--output_file", type=str, help="output file")
        # parser.add_argument("--add_loss", type=bool, default=True, help="when processing features, add loss")
        # parser.add_argument("--add_pct", type=bool, default=True, help="when processing features, add lt_zero_pct")
        # parser.add_argument("--add_std", type=bool, default=True, help="when processing features, add std")
        # parser.add_argument("--add_corr", type=bool, default=True, help="when processing features, add corr")

        parser.add_argument("--do_normalize", action="store_true", help="normalize the features")
        return parser.parse_args()
    
    @classmethod
    def process_features(cls, input_file, output_file, do_normalize=False):
        """
        Process features from raw features.

            raw_features: {losses, begin_idx_list, ll_tokens_list, label_int, label, text}
            ==>
            processed_features: {values, label_int, label}

            values = {losses, lt_zero_percents, std_deviations, pearson_list, spearmann_list}
        """

        # jsonl read
        with open(input_file, 'r') as f:
            raw_features = [json.loads(line) for line in f.readlines()]
        
        print('input file:{}, length:{}'.format(input_file, len(raw_features)))

        with open(output_file, 'w', encoding='utf-8') as f:
            for raw_feature in tqdm(raw_features):
                losses = raw_feature['losses']
                begin_idx_list = raw_feature['begin_idx_list']
                ll_tokens_list = raw_feature['ll_tokens_list']
                
                # un-nest ll_tokens_list (JURY RIGGED)
                ll_tokens_list =  list(itertools.chain.from_iterable(ll_tokens_list))
                label_int = raw_feature['label_int']
                label = raw_feature['label']
                text = raw_feature['text']


                # losses, begin_idx_list, ll_tokens_list, label_int, label = raw_feature
                try:
                    # ll_tokens_len_list = [len(ll_tokens) for ll_tokens in ll_tokens_list]
                    # if ll_tokens_len_list.count(ll_tokens_len_list[0]) != len(ll_tokens_len_list):
                    #     print(ll_tokens_len_list)

                    # Align all vectors in ll_tokens_list
                    # ll_tokens_list = np.array(ll_tokens_list)
                    begin_idx_list = np.array(begin_idx_list)
                    # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
                    max_begin_idx = np.max(begin_idx_list)
                    # Truncate all vectors
                    if len(ll_tokens_list) > 0 and max_begin_idx < len(ll_tokens_list[0]):
                        for idx, ll_tokens in enumerate(ll_tokens_list):
                            ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
                            
                    # ll_tokens_list = ll_tokens_list[:, max_begin_idx:]

                    # Get the length of all vectors and take the minimum
                    min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
                    # Align the lengths of all vectors
                    for idx, ll_tokens in enumerate(ll_tokens_list):
                        ll_tokens_list[idx] = ll_tokens[:min_len]
                        if len(ll_tokens)==0 or min_len == 0:
                            print("problem aligning list for sample. min token list length is zero")
                    # ll_tokens_list = ll_tokens_list[:, :min_len]

                    if do_normalize:
                        # print("normalize: {}".format(do_normalize))
                        # Normalize using L1 normalization
                        ll_tokens_list_normalized = normalize(ll_tokens_list, norm='l1')
                        # Convert back to Python lists
                        lls = ll_tokens_list_normalized.tolist()
                    else:
                        # print("normalize: {}".format(do_normalize))
                        lls = ll_tokens_list


                except Exception as e:
                    """
                    [0, 0, 0, 0], too short, discard this sample
                    """
                    print(e)
                    print("fail to process this sample, discard it, text:{}".format(text))
                    print()
                    continue

                try:
                    lt_zero_percents = []
                    std_deviations = []
                    deviations = []
                    pearson_list = []
                    spearmann_list = []
                    
                    for i in range((len(lls))):
                        for j in range(i + 1, len(lls)):
                            # lls[i], ll[j]
                            deviation_ij = [li - lj for li, lj in zip(lls[i], lls[j])]
                            # `lt` means `less than`
                            deviation_lt_zero_ij = [d < 0 for d in deviation_ij]
                            lt_zero_pct_ij = sum(deviation_lt_zero_ij) / len(
                                deviation_lt_zero_ij)
                            std_ij = np.std(deviation_ij)
                            lt_zero_percents.append(lt_zero_pct_ij)
                            std_deviations.append(std_ij)
                            deviations.append(deviation_ij)
                            pearson = scipy.stats.pearsonr(lls[i], lls[j])[0]
                            spearmann = scipy.stats.spearmanr(lls[i], lls[j])[0]

                            pearson_list.append(pearson)
                            spearmann_list.append(spearmann)

                    values = {'losses': losses,
                            'lt_zero_percents': lt_zero_percents,
                            'std_deviations': std_deviations,
                            'pearson_list': pearson_list,
                            'spearmann_list': spearmann_list}

                    processed_feature = {'values': values,
                                        'label_int': label_int,
                                        'label': label,
                                        'text': text}

                    f.write(json.dumps(processed_feature, ensure_ascii=False) + '\n')
                except:
                    print("fail may due to speraman or pearson")
                    print(text)
                    print(lls[i], lls[j])


if __name__ == "__main__":
    args = SeqXFeatureProcessor.parse_args()    
    feature_processor = SeqXFeatureProcessor()
    
    print(args.do_normalize)
    feature_processor.process_features(args.input_file, args.output_file, args.do_normalize)
