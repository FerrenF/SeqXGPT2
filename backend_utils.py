import re
import torch
import numpy as np
import unicodedata


def _split_en_sentence(sentence, use_sp=False):
    r"""
    CHANGES: Unmodified (09/24)
    Split an English sentence into a sequence of words and whitespace characters according to whitespace characters.

    Args:
        use_sp(`bool`, defaults to `False`): 
            Whether or not based on the SentencePiece Algorithm.
            When using the SentencePiece algorithm, it's necessary to replace ' ' with '▁' during the processing of the sentence.

    For example: 
    sentence = 'I am Ironman.'
    words = split_en_sentence(sentence)
    print(words)
    ['I', ' ', 'am', ' ', 'Ironman.']

    sentence = 'I am Ironman.'
    words = split_en_sentence(sentence, use_sp=True)
    print(words)
    ['I', '▁', 'am', '▁', 'Ironman.']
    """
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words

def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    """
    CHANGED: Removed cn_split_sentence
    """
    return _split_en_sentence(sentence, use_sp)



""" Learn more about BytePair Encoders (BPE) at huggingface:
        https://huggingface.co/learn/nlp-course/chapter6/5
"""


class tokenizerCommon:
    
    @staticmethod
    def calc_sent_ppl(outputs, labels):
        r"""
        Changes: Annotated
        
        :param outputs: The output of the language model, typically containing logits (the raw, unnormalized scores the model assigns to each token in the vocabulary for each position in the input).
        :param labels:  The actual token IDs (correct output tokens) that the model is supposed to predict. These are the ground truth labels for computing loss.
        :return: sentence ppl based on loss, list of subtoken ppl based on loss.
        
        
        The logit function is a crucial concept in statistics and machine learning, particularly within the field of logistic regression. 
        It serves as a link function that maps probabilities ranging between 0 and 1 to real numbers on the entire number line, which can then be used to express linear relationships.
        In essence, the logit function is the inverse of the logistic sigmoid function and is used to model the odds of a binary outcome.
        https://deepai.org/machine-learning-glossary-and-terms/logit
        
        Squeeze is a numpy method that removes axes of length one from a.
        """
        
        lm_logits = outputs.logits.squeeze()  # seq-len, Voutputs.logits is a tensor of shape (batch_size, seq_len, vocab_size). squeeze() function removes any dimensions of size 1 (e.g., if batch_size = 1), simplifying the tensor to shape (seq_len, vocab_size).
        shift_logits = lm_logits[..., :-1, :].contiguous() # removes the last token from the logits, as we are predicting the next token at each step, and we don't need to predict after the final token in the sequence.
        shift_labels = labels[..., 1:].contiguous() # shifts the labels so that we're comparing the logits from position i to the actual label from position i + 1. This ensures that each token is predicted based on the previous ones.
        
        # Cross-entropy loss measures how far the predicted distribution (logits) is from the actual distribution (labels). It computes the log-likelihood of the correct token being predicted at each position.
        loss_func = torch.nn.CrossEntropyLoss(reduction='none') #reduction='none' ensures a loss for each token, rather than a single scalar
        
        # computes the loss between the shifted logits and labels. shift_labels.view(-1) flattens the labels into a 1D vector to match the expected shape by PyTorch.
        ll = loss_func(shift_logits, shift_labels.view(-1))
        
        loss = ll.mean().item() # calculates the average loss across all tokens in the sentence, which is used to compute sentence-level perplexity. The mean loss is a scalar value that represents the model's overall confidence.
        ll = ll.tolist() # converts the token-level loss values to a list
        return loss, ll
    
    @staticmethod
    def get_bbpe_bytes(pplCalcBase, words):
        """
        CHANGES: Unmodified (09/24)
        """
        bbs = []  # bbpe_bytes. This will store the byte-pair encoded (BBPE) version of the words.
        bbs_to_words = [] # this will map each byte from the bbs list back to its original word, associating bytes with word indices.
        for idx, word in enumerate(words):
            """For each word in words:
                1. Convert the word into unicode with word.encode
                2. Simultaneously look up the byte-pair encoding for each byte in the self.byte_encoder didtionary
                3. Add encoded bytes to the bbs array with bbs.extend()
            """
            byte_list = [pplCalcBase.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            # Also, add the index of the word (idx) to the bbs_to_words array, and do it in the amount of len(byte_list) times to account for the number of bytes.
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words
    
    @staticmethod
    def calc_token_ppl(bbs_to_words, bbs_ll):
        """
        CHANGES: Annotated, moved into static method in common class
        
        calculates the perplexity of tokens based on byte-level perplexity values (bbs_ll) and a mapping of byte indices to word indices (bbs_to_words)
        
        :param bbs_to_words: mapping of byte indices to word indices.
        :param bbs_ll: list of byte-level perplexity values.
        :return: list of token ppl.
        """
        start = 0 # the starting index for the current group of bytes that belong to the same token.
        ll_tokens = [] # holds the final token-level perplexities (averaged from the byte-level perplexities).
        while start < len(bbs_to_words) and start < len(bbs_ll):
            
            """         Main Loop
                    - iterates over the entire bbs_to_words and bbs_ll lists, processing one token's bytes at a time
            """
            end = start + 1
            while   end < len(bbs_to_words)   and     bbs_to_words[end] == bbs_to_words[start]:
                # this loop continues incrementing `end` until it finds a byte that belongs to a different token (i.e., bbs_to_words[end] != bbs_to_words[start]).
                end += 1
                
            if end > len(bbs_ll):
                break
            ll_token = bbs_ll[start:end]
            ll_tokens.append(np.mean(ll_token)) # calculates the mean of the byte-level log-likelihoods for the current token and appends it to the ll_tokens list.
            start = end
            
        return ll_tokens
    
import openai
class TikTokenizerPPLCalc(object):
    """ base_tokenizer is based on the 'BBPE Algorithm' """

    def __init__(self, base_model, base_tokenizer):
        self.base_model = base_model # A pretrained openai model
        self.base_tokenizer = base_tokenizer # A base tokenizer

    def get_bbs_ll(self, tokens, ll):
        r"""
        This method computes byte-level log-likelihoods from token log-likelihoods.
        :param tokens: The tokens generated by OpenAI's API.
        :param ll: The log-likelihoods corresponding to the tokens.       
        :return bbs_ll: list of bbpe_byte's ll.
          If the token is a byte-encoded token (e.g., bytes:\x), the method counts how many byte sequences are present and adds a placeholder log-likelihood for each byte (since GPT models output tokens, not bytes).
          If it's a special token (e.g., <pad>, <eos>), it handles it as a single unit.                
          Otherwise, it breaks the token down into its UTF-8 encoded bytes.
        """
        
        bbs_ll = []
        for idx, token in enumerate(tokens):
           
            if token.startswith('bytes:'):
                byte_list = [0 for i in range(token.count('\\x'))]
            elif token in self.base_tokenizer.special_tokens_set:
                byte_list = [token]
            else:
                byte_list = [b for b in token.encode("utf-8")]
            bbs_ll.extend([ll[idx] for _ in range(len(byte_list))])
        return bbs_ll

    def get_begin_word_idx(self, tokens, bbs_to_words):
        r"""
            This method decodes the first token to determine how many bytes it contains and uses that to find where the next word starts in the bbs_to_words list.
            
            :param tokens: The list of tokens from OpenAI.
            :param bbs_to_words: A list mapping byte indices back to the words they belong to.
            :return begin_word_idx: index of the first word after a sequence of bytes.
        """
        token = tokens[0]
        if token.startswith('bytes:'):
            byte_list = [0 for i in range(token.count('\\x'))]
        elif token in self.base_tokenizer.special_tokens_set:
                byte_list = [token]
        else:
            byte_list = [b for b in token.encode("utf-8")]
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        
        r"""
        
            WARNINGS: openai.Completions is a legacy api endpoint. 
            
            The main method responsible for interacting with the OpenAI API to compute perplexity.
            
            :return loss: the average loss (perplexity)
            :return begin_word_idx: the index of the beginning word after byte-level tokens
            :return ll_tokens: the log-likelihoods for each token
        """
        
        # First, split the text.
        words = split_sentence(text)
        bbs, bbs_to_words = tokenizerCommon.get_bbpe_bytes(self, words)

        # The OpenAI API is called using the Completion.create() function. 
        # This generates a response based on the prompt (the input text). The model's log probabilities (logprobs) are returned along with the tokens.
    
        res = openai.Completion.create(model=self.base_model,
                                       prompt=text,
                                       max_tokens=0,
                                       temperature=1,
                                       top_p=1,
                                       logprobs=5,
                                       echo=True) # 'echo=True' This parameter tells the model to return the input tokens and their associated log probabilities.
        
        r"""
            The API will return:
                logprobs['token_logprobs']: Log probabilities of the tokens.
                logprobs['tokens']: The actual tokens generated by the model.
        """
        res = res['choices'][0]
        
        
        token_logprobs = res['logprobs']['token_logprobs']
        token_logprobs[0] = 0 # The log-likelihood for the first token is set to 0 because no prediction is made for the first token in sequence models.
        ll = [-logprob for logprob in token_logprobs] # The log probabilities are converted to negative log-likelihoods (since perplexity is calculated from negative log-likelihoods).
        tokens = res['logprobs']['tokens']
        loss = np.mean(ll[1:])

        bbs_ll = self.get_bbs_ll(tokens, ll) # byte-level log-likelihoods are calculated using get_bbs_ll
        ll_tokens = tokenizerCommon.calc_token_ppl(bbs_to_words, bbs_ll) #  token-level perplexities are computed using calc_token_ppl
        
        begin_word_idx = self.get_begin_word_idx(tokens, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens]
    

    def calc_ppl(self, text, token_logprobs, tokens):
        words = split_sentence(text)
        bbs, bbs_to_words = tokenizerCommon.get_bbpe_bytes(self, words)

        token_logprobs[0] = 0
        ll = [-logprob for logprob in token_logprobs]
        loss = np.mean(ll[1:])

        bbs_ll = self.get_bbs_ll(tokens, ll)
        ll_tokens = tokenizerCommon.calc_token_ppl(bbs_to_words, bbs_ll)
        begin_word_idx = self.get_begin_word_idx(tokens, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens]


class BBPETokenizerPPLCalc(object):
    """ base_tokenizer is based on the 'BBPE Algorithm' """

    def __init__(self, byte_encoder, base_model, base_tokenizer, device):
        self.byte_encoder = byte_encoder
        self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_bbs_ll(self, input_ids, ll):
        r"""
        CHANGES: Replaced _conver_id_to_token with updated method convert_ids_to_tokens, Annotated                    
        :return bbs_ll: a list where each byte of the byte-pair encoded tokens has a corresponding log-likelihood value.
        """
        input_ids = input_ids.squeeze() # Remove dimensions of size 1 from the tensor.
        
        tokenized_tokens = self.base_tokenizer.convert_ids_to_tokens(input_ids)
        bbs_ll = [] # holds the final list of log-likelihood values, corresponding to each byte of the sub-tokens
        
        
        # The first token in GPT-2 is often special because GPT-2 doesn’t include a start-of-sequence token (<s>). 
        # This means the log-likelihood for the first token is not calculated.
        byte_list = [self.byte_decoder[c] for c in tokenized_tokens[0]]        
        # Instead of calculating, set the loss to 0 for each byte of the first token: 
        bbs_ll.extend([0 for i in range(len(byte_list))])
        
        # For the remaining tokens:
        for idx, token in enumerate(tokenized_tokens[1:]):
            byte_list = [self.byte_decoder[c] for c in token]
            bbs_ll.extend(ll[idx] for i in range(len(byte_list))) # extend bbs_ll by repeating the loss value ll[idx] (the log-likelihood for that token) as many times as there are bytes in the toke
        return bbs_ll

    def get_begin_word_idx(self, input_ids, bbs_to_words):
        """
        Changes: Replaced _conver_id_to_token with updated method convert_ids_to_tokens 
        """
        input_ids = input_ids.squeeze() # Remove dimensions of size 1 from the tensor.
        begin_token = self.base_tokenizer.convert_ids_to_token(input_ids[0])
        byte_list = [self.byte_decoder[c] for c in begin_token]
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        
        r"""
        Compute the perplexity of a given text using a pre-trained language model from Hugging Face's transformers library.
        Breaks the process into several steps, including tokenizing the input text, calculating the sentence and token-level perplexity,
        and returning detailed information on how well the model predicts the tokens.
        https://huggingface.co/docs/transformers/main_classes/tokenizer
        
        
        Changes: annotated
        
        :param text: Input text for tokenization and perplexity extraction
        :return loss: the loss, begin_word_idx: word indices, ll_tokens: and token perplexities:.
        """
        tokenized = self.base_tokenizer(text,
                                        return_tensors="pt").to(self.device) # pt specifies a pytorch tensor object as a return type. '.to(self.device)' moves the tensor to the appropriate device (GPU or CPU).
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ] #Sets a sequence limit of 1024 tokens.
        labels = labels[:, :1024, ] #Sets a sequence limit of 1024 tokens.
        
        
        words = split_sentence(text) # splits the input text into a list of words.
        bbs, bbs_to_words = tokenizerCommon.get_bbpe_bytes(self,words) # performs Byte-Pair Encoding (BPE) to get the byte-level tokens (bbs) and the mapping from bytes to words (bbs_to_words).
 
        outputs = self.base_model(input_ids=input_ids, labels=labels) # the input_ids and labels to the pre-trained model
        
        # see the annotations above for a description of these methods
        loss, ll = tokenizerCommon.calc_sent_ppl(outputs, labels) 
        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = tokenizerCommon.calc_token_ppl(bbs_to_words, bbs_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens] # Return the loss, word indices, and token perplexities:


class CharLevelTokenizerPPLCalc(object):
    """ base_tokenizer is based on Char Level """

    def __init__(self, all_special_tokens, base_model, base_tokenizer, device):
        self.all_special_tokens = all_special_tokens # Stores any special tokens (like padding, end-of-sequence, etc.) that the tokenizer might use.
        self.base_model = base_model # the pre-trained model that will be used to generate predictions and calculate perplexity.
        self.base_tokenizer = base_tokenizer # pre-trained character-level tokenizer. 
        self.device = device # CUDA specification

    def get_chars(self, words):
        """
            Changes: Annotated
            
            get_chars is responsible for converting words into a list of individual characters and maintaining a mapping between characters and their corresponding word index
            
            Example:
                For the word list ["hello", "world"]:

                chars = ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd']
                chars_to_words = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (mapping each character back to the word it belongs to)
            
            :param words: A list of words
            :return
                chars: A list that stores the characters in each word.
                chars_to_words: A mapping that relates each character to the index of the word it belongs to. 
        """
        chars = []
        chars_to_words = []
        for idx, word in enumerate(words):
            char_list = list(word)
            chars.extend(char_list)
            chars_to_words.extend([idx for i in range(len(char_list))])
        return chars, chars_to_words
       
    def get_chars_ll(self, input_ids, ll):
        """
        get_chars_ll is designed to calculate the character-level log-likelihoods (LL) by transforming token-level log-likelihoods into character-level ones.
        
        :param input_ids: The token IDs generated by the tokenizer (representing the tokenized input text).
        :param ll: The list of log-likelihoods at the token level, corresponding to the predictions made by the language model.
        
        :return chars_ll: list of char's ll.
        
        
        Example:

            If the input tokens were:
                tokenized_tokens = ["hello", "world", "<pad>"] 
            And the token-level log-likelihoods (ll) were:
                ll = [0.1, 0.2]

            This method should:

                For "hello", break it into ['h', 'e', 'l', 'l', 'o'] and assign the log-likelihood 0.1 to each character.
                For "world", break it into ['w', 'o', 'r', 'l', 'd'] and assign the log-likelihood 0.2 to each character.
                For the special token "<pad>", treat it as a single "character" and assign no log-likelihood to it (since it's padded).

            Thus, the chars_ll returned would be something like:

                [0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0]
        
        
        """
        input_ids = input_ids.squeeze() # we have seen this a few times now, so I won't annotate this method anymore.
        
        tokenized_tokens = []
        for input_id in input_ids:
            # each input_id is decoded back into a text token using the base_tokenizer class. The decoded tokens are then stored in the tokenized_tokens list.
            tokenized_tokens.append(self.base_tokenizer.decode(input_id))
            
            
        chars_ll = []
        # the tokenizer may not include <s> before the sub_tokens, so:
        # If the first token is a special token (e.g., <pad>, <eos>), treat the whole token as a single character.
        # If it's a regular token, break it down into individual characters.
        # for the first token, set the character log-likelihood of each first token to 0
        token = tokenized_tokens[0]
        if token in self.all_special_tokens:
            char_list = [token]
        else:
            char_list = list(token)
            
            
        
        chars_ll.extend([0 for i in range(len(char_list))])
        # next we process the following sequence
        for idx, token in enumerate(tokenized_tokens[1:]):
            
            # once again, determine whether a token is a special token or not, and process as above.
            if token in self.all_special_tokens:
                char_list = [token]
            else:
                char_list = list(token)
            chars_ll.extend(ll[idx] for i in range(len(char_list))) #extend chars_ll with the log-likelihood of the current token (ll[idx]) repeated for each character in the token.
        return chars_ll

    def get_begin_word_idx(self, input_ids, chars_to_words):
        input_ids = input_ids.squeeze()
        begin_token = self.base_tokenizer.decode([input_ids[0]])
        if begin_token in self.all_special_tokens:
            char_list = [begin_token]
        else:
            char_list = list(begin_token)
        begin_word_idx = chars_to_words[len(char_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        tokenized = self.base_tokenizer(text,
                                        return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        words = split_sentence(text)
        chars, chars_to_words = self.get_chars(words)

        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss, ll = tokenizerCommon.calc_sent_ppl(outputs, labels)
        chars_ll = self.get_chars_ll(input_ids, ll)
        ll_tokens = tokenizerCommon.calc_token_ppl(chars_to_words, chars_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, chars_to_words)
        return [loss, begin_word_idx, ll_tokens]


class SPLlamaTokenizerPPLCalc(object):
    """ base_tokenizer is based on the `SentencePiece Algorithm` for Llama models """

    def __init__(self, base_model, base_tokenizer, device):
        # Llama tokenizer has byte level tokens for words which is not in the `tokenizer vocab`
        self.byte_encoder = {i: f'<0x{i:02X}>' for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_sp_bytes(self, words):
        bbs = []  # bytes
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words

    def get_bbs_ll(self, input_ids, ll):
        """
        Changes: changed old method to convert_ids_to_tokens
        
        :return bbs_ll: list of bbpe_byte's ll.
        """
        input_ids = input_ids.squeeze()
        # because `sentencepiece tokenizer` add `<s>▁` before all sentence.
        # this step we remove `<s>`, because it is treated as a separate token.
        input_ids = input_ids[1:]
        tokenized_tokens = self.base_tokenizer.convert_ids_to_tokens(input_ids)
        bbs_ll = []
        for idx, token in enumerate(tokenized_tokens):
            if self.base_tokenizer.sp_model.IsByte(input_ids[idx].item()):
                byte_list = [token]
            else:
                byte_list = [
                    self.byte_encoder[b] for b in token.encode("utf-8")
                ]
            bbs_ll.extend(ll[idx] for i in range(len(byte_list)))
        # because `sentencepiece tokenizer` add `<s>▁` before all sentence.
        # this step we remove `▁`, because it is treated as the first token or part of the first token which corresponds to the first logit.
        bbs_ll = bbs_ll[len('▁'.encode("utf-8")):]
        return bbs_ll

    def forward_calc_ppl(self, text):
        tokenized = self.base_tokenizer(text,
                                        max_length=1024,
                                        truncation=True,
                                        return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        words = split_sentence(text, use_sp=True)
        bbs, bbs_to_words = self.get_sp_bytes(words)

        # Here we don't pass the labels because the output_logits and labels may not on the same device is you not carefully set it.
        outputs = self.base_model(input_ids=input_ids)
        loss, ll = tokenizerCommon.calc_sent_ppl(outputs, labels)
        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = tokenizerCommon.calc_token_ppl(bbs_to_words, bbs_ll)
        # ll_tokens has removed `<s>_`, the first element is the logit of the first word
        begin_word_idx = 0
        return [loss, begin_word_idx, ll_tokens]


class SPChatGLMTokenizerPPLCalc(object):
    """ base_tokenizer is based on the `SentencePiece Algorithm` for ChatGLM models """

    def __init__(self, base_model, base_tokenizer, device):
        # ChatGLM tokenizer has byte level tokens for words which is not in the `tokenizer vocab`
        self.byte_encoder = {i: f'<0x{i:02X}>' for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_sp_bytes(self, words):
        bbs = []  # bytes
        bbs_to_words = []
        for idx, word in enumerate(words):
            word = unicodedata.normalize('NFKC', word)
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words

    def get_bbs_ll(self, input_ids, ll, text):
        """
        :return bbs_ll: list of bbpe_byte's ll.
        """
        input_ids = input_ids.squeeze()
        # because '[gMASK]' and '<sop>' will be added at the end of the sentence.
        input_ids = input_ids[:-2]
        tokenized_tokens = self.base_tokenizer.convert_ids_to_tokens(input_ids)
        bbs_ll = []
        for idx, token in enumerate(tokenized_tokens):
            if token in self.byte_encoder.values():
                byte_list = [token]
            elif token == '<|tab|>':
                byte_list = [token]
            elif token == '<n>':
                byte_list = [token]
            elif token.startswith('<|blank_'):
                num = re.findall(r"(\d+)", token)[0]
                num = int(num)
                byte_list = [
                    self.byte_encoder[b] for b in ('▁' * num).encode("utf-8")
                ]
            else:
                byte_list = [
                    self.byte_encoder[b] for b in token.encode("utf-8")
                ]
            if idx == 0:
                bbs_ll.extend(0 for i in range(len(byte_list)))
            else:
                bbs_ll.extend(ll[idx - 1] for i in range(len(byte_list)))
        # this step we remove `▁`, because it is treated as the first token or part of the first token which corresponds to the first logit.
        if (not text.startswith(' ') and tokenized_tokens[0].startswith('▁')) or \
        (text.startswith('  ') and tokenized_tokens[0].startswith('▁')):
            bbs_ll = bbs_ll[len('▁'.encode("utf-8")):]
        return bbs_ll

    def get_begin_word_idx(self, input_ids, bbs_to_words, text):
        
        """
            Changes: updated old method to convert_ids_to_tokens
        """
        input_ids = input_ids.squeeze()
        begin_token = self.base_tokenizer.convert_ids_to_tokens(input_ids[0].item())
        
        if (not text.startswith(' ') and begin_token.startswith('▁')) or \
        (text.startswith('  ') and begin_token.startswith('▁')):
            begin_token = begin_token[1:]
        token = begin_token
        if len(token) == 0:
            return 0
        if token in self.byte_encoder.values():
            byte_list = [token]
        elif token == '<|tab|>':
            byte_list = [token]
        elif token == '<n>':
            byte_list = [token]
        elif token.startswith('<|blank_'):
            num = re.findall(r"(\d+)", token)[0]
            num = int(num)
            byte_list = [
                self.byte_encoder[b] for b in ('▁' * num).encode("utf-8")
            ]
        else:
            byte_list = [self.byte_encoder[b] for b in token.encode("utf-8")]
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        tokenized = self.base_tokenizer(text,
                                        return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        words = split_sentence(text, use_sp=True)
        bbs, bbs_to_words = self.get_sp_bytes(words)

        # Here we don't pass the labels because the output_logits and labels may not on the same device is you not carefully set it.
        outputs = self.base_model(input_ids=input_ids)
        loss, ll = tokenizerCommon.calc_sent_ppl(outputs, labels)
        bbs_ll = self.get_bbs_ll(input_ids, ll, text)
        ll_tokens = tokenizerCommon.calc_token_ppl(bbs_to_words, bbs_ll)
        # ll_tokens has removed `<s>_`, the first element is the logit of the first word
        begin_word_idx = self.get_begin_word_idx(input_ids, bbs_to_words, text)
        return [loss, begin_word_idx, ll_tokens]