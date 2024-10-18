__all__ = [
    "ConditionalRandomField",
    "allowed_transitions"
]

from typing import Union, List, Tuple

import torch
from torch import nn

from collections import Counter
from functools import partial
from functools import wraps
from typing import List, Callable, Union
import io

import logging
logger = logging.getLogger(__name__)

def _is_iterable(value):
    try:
        iter(value)
        return True
    except BaseException as e:
        return False
    
class Option(dict):
    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.pop(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


class VocabularyOption(Option):
    """

    """
    def __init__(self,
                 max_size=None,
                 min_freq=None,
                 padding='<pad>',
                 unknown='<unk>'):
        super().__init__(
            max_size=max_size,
            min_freq=min_freq,
            padding=padding,
            unknown=unknown
        )


def _check_build_vocab(func: Callable):
    r"""
    A decorator to make sure the indexing is built before used.

    :param func: 传入的callable函数

    """    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self._word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)
    
    return _wrapper


def _check_build_status(func):
    r"""
    A decorator to check whether the vocabulary updates after the last build.

    :param func: 用户传入要修饰的callable函数

    """    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.rebuild is False:
            self.rebuild = True
            if self.max_size is not None and len(self.word_count) >= self.max_size:
                logger.warning("Vocabulary has reached the max size {} when calling {} method. "
                            "Adding more words may cause unexpected behaviour of Vocabulary. ".format(
                    self.max_size, func.__name__))
        return func(self, *args, **kwargs)
    
    return _wrapper


class Vocabulary(object):
    r"""
    For building, storing and using a one-to-one mapping from `str` to `int`::

        from fastNLP.core import Vocabulary
        vocab = Vocabulary()
        word_list = "this is a word list".split()
        # vocab updates its own dictionary, input is a list
        vocab.update(word_list)
        vocab["word"] # str to int
        vocab.to_word(5) # int to str

    :param max_size: `Vocabulary` The maximum size, i.e. the maximum number of words that can be stored
If ``None``, there is no size limit.
    :param min_freq: The minimum frequency of a word in the text to be recorded, should be greater than or equal to 1.
If it is less than this frequency, the word will be considered `unknown`. If it is ``None``, all words in the text will be recorded.
    :param padding: Padding characters. If set to ``None`` ,
padding is not considered in the vocabulary, nor is it counted in the vocabulary size. ``None`` is mostly used when creating a vocabulary for a label.
    :param unknown: unknown character, all unrecorded words will be considered as `unknown` when converted to :class:`int`.
If set to ``None``, `unknown` is not considered in the vocabulary and is not counted in the vocabulary size.
``None`` is mostly used when building a vocabulary for a label
    """

    def __init__(self, max_size:int=None, min_freq:int=None, padding:str='<pad>', unknown:str='<unk>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        #  Used to carry words that do not require separate entries. For details, see the from_dataset() method
        self._no_create_word = Counter()

    @property
    @_check_build_vocab
    def word2idx(self):
        return self._word2idx

    @word2idx.setter
    def word2idx(self, value):
        self._word2idx = value

    @property
    @_check_build_vocab
    def idx2word(self):
        return self._idx2word

    @idx2word.setter
    def idx2word(self, value):
        self._word2idx = value

    @_check_build_status
    def update(self, word_lst: list, no_create_entry:bool=False):
        r"""
       Increase the frequency of occurrence of words in the dictionary in sequence
       
        :param word_lst: Words in list form, such as word_list=['I', 'am', 'a', 'Chinese'], each word in the list will be calculated and added to the dictionary.
        :param no_create_entry: If the word comes from a non-training set, it is recommended to set it to ``True``.
            
            * If it is `` true`` - no this word will create a separate entry, it will always be pointing to the representation of ``<UNK> `` ``
            *If it is ``False`` -- create a separate entry for this word. If the word comes from the validation set or training set, it is generally set to True, if it comes from the training set, it is generally set to ``False``;
              
           There are two cases: If a new word is added and ``no_create_entry`` is ``True``, but the word is already in the Vocabulary and is not ``no_create_entry``, 
           a separate vector will still be created for the word; if ``no_create_entry`` is ``False``, but the word is already in the Vocabulary and is not ``no_create_entry``,
           then the word will be considered to need to create a separate vector.

        """
        self._add_no_create_entry(word_lst, no_create_entry)
        self.word_count.update(word_lst)
        return self
    
    @_check_build_status
    def add(self, word:str, no_create_entry:bool=False):
        r"""
        Increase the frequency of a new word in the dictionary

        :param word: To add a new word to the dictionary, ``word`` type string
        :param no_create_entry: If the word comes from a non-training set, the recommended setting is ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
              
            There are two situations: If a new word ，and ``no_create_entry`` 为 ``True``, but the word has been used before in Vocabulary and not 
            ``no_create_entry`` a seperate vector vector is created; if ``no_create_entry`` is ``False`` , if the word is already in
            Vocabulary and not ``no_create_entry`` , then this word will be considered to need to create a separate vector

        """
        self._add_no_create_entry(word, no_create_entry)
        self.word_count[word] += 1
        return self
    
    def _add_no_create_entry(self, word:Union[str, List[str]], no_create_entry:bool):
        r"""
        When adding a new word, check the setting of _no_create_word.

        :param word: 要添加的新词或者是 :class:`List`类型的新词，如 word='I' 或者 word=['I', 'am', 'a', 'Chinese'] 均可
        :param no_create_entry: 如果词语来自于非训练集建议设置为 ``True`` 。
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
    
        :return:

        """
        if isinstance(word, str) or not _is_iterable(word):
            word = [word]
        for w in word:
            if no_create_entry and self.word_count.get(w, 0) == self._no_create_word.get(w, 0):
                self._no_create_word[w] += 1
            elif not no_create_entry and w in self._no_create_word:
                self._no_create_word.pop(w)
    
    @_check_build_status
    def add_word(self, word:str, no_create_entry:bool=False):
        r"""
        Increase the frequency of a new word in the dictionary

        :param word: The new word to be added to the dictionary, ``word`` is a string
        :param no_create_entry: If the word comes from a non-training set, it is recommended to set it to ``True``.
            
            * 如果为 ``True`` -- 则不会有这个词语创建一个单独的 entry ，它将一直被指向 ``<UNK>`` 的表示；
            * 如果为 ``False`` -- 为这个词创建一个单独的 entry。如果这个词来自于验证集或训练集，一般设置为 ``True`` ，如果来自于训练集一
              般设置为 ``False``；
              
            有以下两种情况: 如果新加入一个 word ，且 ``no_create_entry`` 为 ``True``，但这个词之前已经在 Vocabulary 中且并不是 
            ``no_create_entry`` 的，则还是会为这个词创建一个单独的 vector ; 如果 ``no_create_entry`` 为 ``False`` ，但这个词之
            前已经在 Vocabulary 中且并不是 ``no_create_entry的`` ，则这个词将认为是需要创建单独的 vector 的。

        """
        self.add(word, no_create_entry=no_create_entry)
    
    @_check_build_status
    def add_word_lst(self, word_lst: List[str], no_create_entry:bool=False):
        r"""
        Add the frequency of words in the sequence in the sequence in the dictionary
        
        :param word_lst: the list sequence of the new word that needs to be added, such as word_LST = ['i', 'am', 'a', 'Chinese 
        :param no_create_entry: If the word comes from the non -training collection, True. If the word comes from the validation set or training set, it is generally set to ``True``, and if it comes from the training set, it is generally set to ``False``;

        There are two cases: If a new word is added and ``no_create_entry`` is ``True``, but the word has been in the Vocabulary before and is not ``no_create_entry``, a separate vector will still be created for the word; If ``no_create_entry`` is ``False``, but the word has been in the Vocabulary before and is not ``no_create_entry``, the word will be considered to need to create a separate vector.

        """
        self.update(word_lst, no_create_entry=no_create_entry)
        return self
    
    def build_vocab(self):
        r"""
        Build a dictionary based on the words that have appeared and their frequency. 
        Note: Repeated construction may change the size of the dictionary, but the words that have been recorded in the dictionary will not change the corresponding :class:`int`
        """
        if self._word2idx is None:
            self._word2idx = {}
            if self.padding is not None:
                self._word2idx[self.padding] = len(self._word2idx)
            if (self.unknown is not None) and (self.unknown != self.padding):
                self._word2idx[self.unknown] = len(self._word2idx)
        
        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self._word2idx is not None:
            words = filter(lambda kv: kv[0] not in self._word2idx, words)
        start_idx = len(self._word2idx)
        self._word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False
        return self
    
    def build_reverse_vocab(self):
        r"""
        Based on the `word to index` dict, construct an `index to word` dict.

        """
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        return self
    
    @_check_build_vocab
    def __len__(self):
        return len(self._word2idx)
    
    @_check_build_vocab
    def __contains__(self, item:str):
        r"""
        Check if the word is recorded

        :param item: the word
        :return: True or False
        """
        return item in self._word2idx
    
    def has_word(self, w:str):
        r"""
        Check if the word is recorded::

            has_abc = vocab.has_word('abc')
            # equals to
            has_abc = 'abc' in vocab

        :param item: Input word of type str
        :return: ``True`` or ``False``
        """
        return self.__contains__(w)
    
    @_check_build_vocab
    def __getitem__(self, w):
        r"""
        Supports getting the index of words directly from the dictionary, for example::

            vocab[w]
        """
        if w in self._word2idx:
            return self._word2idx[w]
        if self.unknown is not None:
            return self._word2idx[self.unknown]
        else:
            raise ValueError("word `{}` not in vocabulary".format(w))
    
    
    @property
    def _no_create_word_length(self):
        return len(self._no_create_word)
      
    def _is_word_no_create_entry(self, word:str):
        r"""
        判断当前的word是否是不需要创建entry的，具体参见from_dataset的说明

        :param word: 输入的str类型的词语
        :return: bool值的判断结果
        """
        return word in self._no_create_word
    
    def to_index(self, w:str):
        r"""
        Convert words to numbers. If the word is not recorded in the dictionary, it will be regarded as `unknown`, if ``unknown=None`` , will throw ``ValueError`` ::

            index = vocab.to_index('abc')
            # equals to
            index = vocab['abc']

        :param w: 需要输入的词语
        :return: 词语 ``w`` 对应的 :class:`int`类型的 index
        """
        return self.__getitem__(w)
    
    @property
    @_check_build_vocab
    def unknown_idx(self):
        r"""
        获得 ``unknown`` 对应的数字.
        """
        if self.unknown is None:
            return None
        return self._word2idx[self.unknown]
    
    @property
    @_check_build_vocab
    def padding_idx(self):
        r"""
        获得 ``padding`` 对应的数字
        """
        if self.padding is None:
            return None
        return self._word2idx[self.padding]
    
    @_check_build_vocab
    def to_word(self, idx: int):
        r"""
        给定一个数字, 将其转为对应的词.

        :param idx:
        :return: ``idx`` 对应的词
        """
        return self._idx2word[idx]
    
    def clear(self):
        r"""
        删除 :class:Vocabulary`` 中的词表数据。相当于重新初始化一下。

        :return: 自身
        """
        self.word_count.clear()
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word.clear()
        return self
    
    def __getstate__(self):
        r"""
        用来从 pickle 中加载 data

        """
        len(self)  # make sure vocab has been built
        state = self.__dict__.copy()
        # no need to pickle _idx2word as it can be constructed from _word2idx
        del state['_idx2word']
        return state
    
    def __setstate__(self, state):
        r"""
        支持 pickle 的保存，保存到 pickle 的 data state

        """
        self.__dict__.update(state)
        self.build_reverse_vocab()
    
    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])
    
    @_check_build_vocab
    def __iter__(self):
        # 依次(word1, 0), (word1, 1)
        for index in range(len(self._word2idx)):
            yield self.to_word(index), index

    def save(self, filepath: Union[str, io.StringIO]):
        r"""
        保存当前词表。

        :param filepath: 词表储存路径
        """
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'w', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `path`.")

        f.write(f'max_size\t{self.max_size}\n')
        f.write(f'min_freq\t{self.min_freq}\n')
        f.write(f'unknown\t{self.unknown}\n')
        f.write(f'padding\t{self.padding}\n')
        f.write(f'rebuild\t{self.rebuild}\n')
        f.write('\n')
        # idx: 如果idx为-2, 说明还没有进行build; 如果idx为-1，说明该词未编入
        # no_create_entry: 如果为1，说明该词是no_create_entry; 0 otherwise
        # word \t count \t idx \t no_create_entry \n
        idx = -2
        for word, count in self.word_count.items():
            if self._word2idx is not None:
                idx = self._word2idx.get(word, -1)
            is_no_create_entry = int(self._is_word_no_create_entry(word))
            f.write(f'{word}\t{count}\t{idx}\t{is_no_create_entry}\n')
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()

    @staticmethod
    def load(filepath: Union[str,io.StringIO]):
        r"""
        Load data from file path

        :param filepath: path to read vocabulary
        :return: read :class:`Vocabulary`
        """
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'r', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `path`.")

        vocab = Vocabulary()
        for line in f:
            line = line.strip('\n')
            if line:
                name, value = line.split()
                if name in ('max_size', 'min_freq'):
                    value = int(value) if value!='None' else None
                    setattr(vocab, name, value)
                elif name in ('unknown', 'padding'):
                    value = value if value!='None' else None
                    setattr(vocab, name, value)
                elif name == 'rebuild':
                    vocab.rebuild = True if value=='True' else False
            else:
                break
        word_counter = {}
        no_create_entry_counter = {}
        word2idx = {}
        for line in f:
            line = line.strip('\n')
            if line:
                parts = line.split('\t')
                word,count,idx,no_create_entry = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                if idx >= 0:
                    word2idx[word] = idx
                word_counter[word] = count
                if no_create_entry:
                    no_create_entry_counter[word] = count

        word_counter = Counter(word_counter)
        no_create_entry_counter = Counter(no_create_entry_counter)
        if len(word2idx)>0:
            if vocab.padding:
                word2idx[vocab.padding] = 0
            if vocab.unknown:
                word2idx[vocab.unknown] = 1 if vocab.padding else 0
            idx2word = {value:key for key,value in word2idx.items()}

        vocab.word_count = word_counter
        vocab._no_create_word = no_create_entry_counter
        if word2idx:
            vocab._word2idx = word2idx
            vocab._idx2word = idx2word
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()
        return vocab
    
#class Vocabulary:
#    def __init__(self, idx2word, padding='<pad>', unknown='<unk>'):
#        self.idx2word = idx2word
#        self.padding = padding
#        self.unknown = unknown

def _check_tag_vocab_and_encoding_type(tag_vocab: Union[Vocabulary, dict], encoding_type: str):
    r"""
    Check if the tag in vocab matches the encoding_type

    :param tag_vocab: supports passing in tag Vocabulary; or passing in a dict in the form of {0:"O", 1:"B-tag1"}, i.e. index first and tag last.
    :param encoding_type: bio, bmes, bioes, bmeso
    :return:
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')  # 删除该值
    if tags:  # 如果不为空，说明出现了未使用的tag
        logger.warning(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


def _get_encoding_type_from_tag_vocab(tag_vocab: Union[Vocabulary, dict]) -> str:
    r"""
    
    Given a Vocabulary y, automatically determine the encoding type, supporting bmes, bioes, bmeso, bio
    :param tag_vocab: supports passing in tag Vocabulary; or passing in a dict in the form of {0:"O", 1:"B-tag1"}, i.e. index first, tag second.
    
    :return:
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")




def allowed_transitions(tag_vocab:Union[Vocabulary, dict], encoding_type:str=None, include_start_end:bool=False) -> List[Tuple[int, int]]:
    if encoding_type is None:
        encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)
    else:
        encoding_type = encoding_type.lower()
        _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)

    pad_token = '<pad>'
    unk_token = '<unk>'

    if isinstance(tag_vocab, Vocabulary):
        id_label_lst = list(tag_vocab.idx2word.items())
        pad_token = tag_vocab.padding
        unk_token = tag_vocab.unknown
    else:
        id_label_lst = list(tag_vocab.items())

    num_tags = len(tag_vocab)
    start_idx = num_tags
    end_idx = num_tags + 1
    allowed_trans = []
    if include_start_end:
        id_label_lst += [(start_idx, 'start'), (end_idx, 'end')]
    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        if from_label in [pad_token, unk_token]:
            continue
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            if to_label in [pad_token, unk_token]:
                continue
            to_tag, to_label = split_tag_label(to_label)
            if _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))
    return allowed_trans


def _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):

    if to_tag == 'start' or from_tag == 'end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type == 'bio':
        r"""
        The first row is to_tag, the first column is from_tag. y can be converted under any conditions, - can only be converted when the labels are the same, n cannot be converted
        +-------+---+---+---+-------+-----+
        |       | B | I | O | start | end |
        +-------+---+---+---+-------+-----+
        |   B   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   I   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   O   | y | n | y | n     | y   |
        +-------+---+---+---+-------+-----+
        | start | y | n | y | n     | n   |
        +-------+---+---+---+-------+-----+
        | end   | n | n | n | n     | n   |
        +-------+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ('b', 'o')
        elif from_tag in ['b', 'i']:
            return any([to_tag in ['end', 'b', 'o'], to_tag == 'i' and from_label == to_label])
        elif from_tag == 'o':
            return to_tag in ['end', 'b', 'o']
        else:
            raise ValueError("Unexpect tag {}. Expect only 'B', 'I', 'O'.".format(from_tag))

    elif encoding_type == 'bmes':
        r"""
        The first row is to_tag, the first column is from_tag, y can be converted under any conditions, - can only be converted when the labels are the same, n cannot be converted
        +-------+---+---+---+---+-------+-----+
        |       | B | M | E | S | start | end |
        +-------+---+---+---+---+-------+-----+
        |   B   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   M   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   E   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        |   S   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        | start | y | n | n | y |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |  end  | n | n | n | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ['b', 's']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's']:
            return to_tag in ['b', 's', 'end']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S'.".format(from_tag))
    elif encoding_type == 'bmeso':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S', 'O'.".format(from_tag))
    elif encoding_type == 'bioes':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag == 'i':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'I', 'E', 'S', 'O'.".format(from_tag))
    else:
        raise ValueError("Only support BIO, BMES, BMESO, BIOES encoding type, got {}.".format(encoding_type))


class ConditionalRandomField(nn.Module):
    r"""
    Conditional random fields. 
    
    Provides two methods: 
    
    :meth:`forward` and 
    :meth:`viterbi_decode`, for **training** and **inference** respectively.

    :param num_tags: the number of tags
    :param include_start_end_trans: whether to consider each tag as the start and end score.
    :param allowed_transitions: the internal ``Tuple[from_tag_id(int), to_tag_id(int)]`` is considered as allowed transitions, 
    and other transitions not included are considered forbidden transitions, which can be obtained through the :func:`allowed_transitions` 
    function; if it is ``None``, all transitions are legal.
    
    """

    def __init__(self, num_tags:int, include_start_end_trans:bool=False, allowed_transitions:List=None):
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            has_start = False
            has_end = False
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
                if from_tag_id==num_tags:
                    has_start = True
                if to_tag_id==num_tags+1:
                    has_end = True
            if not has_start:
                constrain[num_tags, :].fill_(0)
            if not has_end:
                constrain[:, num_tags+1].fill_(0)
        self._constrain = nn.Parameter(constrain, requires_grad=False)

    def _normalizer_likelihood(self, logits, mask):
        r"""Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.

        :param logits:FloatTensor, ``[max_len, batch_size, num_tags]``
        :param mask:ByteTensor, ``[max_len, batch_size]``
        :return:FloatTensor, ``[batch_size,]``
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha = alpha + self.start_scores.view(1, -1)

        flip_mask = mask.eq(False)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                    alpha.masked_fill(mask[i].eq(True).view(batch_size, 1), 0)

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(self, logits, tags, mask):
        r"""
        Compute the score for the gold path.
        :param logits: FloatTensor, ``[max_len, batch_size, num_tags]``
        :param tags: LongTensor, ``[max_len, batch_size]``
        :param mask: ByteTensor, ``[max_len, batch_size]``
        :return:FloatTensor, ``[batch_size.]``
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        mask = mask.eq(True)
        flip_mask = mask.eq(False)
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score = score + st_scores + ed_scores
        # return [B,]
        return score

    def forward(self, feats: "torch.FloatTensor", tags: "torch.LongTensor", mask: "torch.ByteTensor") -> "torch.FloatTensor":
        r"""
        Used to calculate the forward loss of ``CRF``. The return value is a :class:`torch.FloatTensor` of shape ``[batch_size,]``. You may need :func:`mean` to find the loss.
            :param feats: feature matrix, shape is ``[batch_size, max_len, num_tags]``
            :param tags: tag matrix, shape is ``[batch_size, max_len]``
            :param mask: shape is ``[batch_size, max_len]``. Positions with **0** are considered padding.
        :return: ``[batch_size,]``
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)

        return all_path_score - gold_path_score

    def viterbi_decode(self, logits: "torch.FloatTensor", mask: "torch.ByteTensor", unpad=False):
        r"""
        
        Given a **feature matrix** and a **transition score matrix**, calculate the best path and the corresponding score

        :param logits: feature matrix, shape is ``[batch_size, max_len, num_tags]``
        :param mask: label matrix, shape is ``[batch_size, max_len]``, **0** positions are considered padding. If it is ``None``, it is considered that there is no padding.
        :param unpad: whether to delete padding from the result:

        - When it is ``False``, the returned value is a tensor of ``[batch_size, max_len]``
        - When it is ``True``, the returned value is :class:`List` [:class:`List` [ :class:`int` ]], the internal :class:`List` [:class:`int` ] is the label of each
        sequence, and the pad part has been removed, that is, each The length of :class:`List` [ :class:`int` ] is the effective length of this sample.
        :return: (paths, scores).

        - ``paths`` -- decoded paths, whose values ​​refer to the ``unpad`` parameter.

        - ``scores`` -- :class:`torch.FloatTensor`, with shape ``[batch_size,]``, corresponding to the score of each optimal path.

        """
        batch_size, max_len, n_tags = logits.size()
        seq_len = mask.long().sum(1)
        logits = logits.transpose(0, 1).data  # L, B, H
        mask = mask.transpose(0, 1).data.eq(True)  # L, B
        flip_mask = mask.eq(False)

        # dp
        vpath = logits.new_zeros((max_len, batch_size, n_tags), dtype=torch.long)
        vscore = logits[0]  # bsz x n_tags
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags + 1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]

        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        end_trans_score = transitions[:n_tags, n_tags+1].view(1, 1, n_tags).repeat(batch_size, 1, 1) # bsz, 1, n_tags

        # For sentences of length 1
        vscore += transitions[:n_tags, n_tags+1].view(1, n_tags).repeat(batch_size, 1) \
            .masked_fill(seq_len.ne(1).view(-1, 1), 0)
        for i in range(1, max_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = logits[i].view(batch_size, 1, n_tags) + trans_score
            score = prev_score + cur_score.masked_fill(flip_mask[i].view(batch_size, 1, 1), 0)  # bsz x n_tag x n_tag
            # It is necessary to consider that the current position is the last one in the sequence
            score += end_trans_score.masked_fill(seq_len.ne(i+1).view(-1, 1, 1), 0)

            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            # Since the final step is to backtrack through last_tags, it is necessary to keep the vscore status of each position
            vscore = best_score.masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                     vscore.masked_fill(mask[i].view(batch_size, 1), 0)

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(max_len, dtype=torch.long, device=logits.device)
        lens = (seq_len - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % max_len

        ans = logits.new_empty((max_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(max_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, max_len in enumerate(lens):
                paths.append(ans[idx, :max_len + 1].tolist())
        else:
            paths = ans
        return paths, ans_score