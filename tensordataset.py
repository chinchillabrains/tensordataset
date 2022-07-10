import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

class Dataset:
    _data_items = []
    _data_labels = []
    _items_vocab = []

    def __init__(self, data) -> None:
        tf.Tensor
        if type(data) != list:
            data = list(data)
        
        self._data_items = tf.data.Dataset.from_tensor_slices(
            list(map(lambda cols: cols[0], data))
        )
        self._data_labels = tf.data.Dataset.from_tensor_slices(
            list(map(lambda cols: cols[1], data))
        )
        
    def get_items(self):
        return self._data_items

    def get_labels(self):
        return self._data_labels

    def build_vocab(self):
        bert_tokenizer_params=dict(lower_case=True)
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size = 80000,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )

        self._items_vocab = bert_vocab.bert_vocab_from_dataset(
            self._data_items.batch(1000).prefetch(3),
            **bert_vocab_args
        )

    def get_vocab(self):
        return self._items_vocab
