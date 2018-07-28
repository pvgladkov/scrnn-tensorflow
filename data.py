import numpy as np


class Vocabulary:
    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len
        self.chars = sorted(list(set(text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))


class DataProvider:

    def __init__(self, text, seq_length, batch_size, logger):
        self.vocab = Vocabulary(text, seq_length)
        self.logger = logger
        self.pointer = 0
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.sentences = []
        self.next_chars = []
        self.x_batches = None
        self.y_batches = None
        self.num_batches = None
        self._load()
        self._create_batches()

    def _x_y(self, sentences, next_chars):
        X = np.zeros((len(sentences), self.seq_length, len(self.vocab.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.vocab.chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.vocab.char_indices[char]] = 1
            y[i, self.vocab.char_indices[next_chars[i]]] = 1
        return X, y

    def _load(self):
        max_len = self.vocab.max_len
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(self.vocab.text) - max_len, step):
            sentences.append(self.vocab.text[i: i + max_len])
            next_chars.append(self.vocab.text[i + max_len])
        self.sentences = sentences
        self.next_chars = next_chars

    def get_data(self):
        return self._x_y(self.sentences, self.next_chars)

    @property
    def vocab_size(self):
        return len(self.vocab.chars)

    def next_batch(self):
        x, y = self._x_y(self.x_batches[self.pointer], self.y_batches[self.pointer])
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def _create_batches(self):
        self.num_batches = int(len(self.sentences) / (self.batch_size * self.seq_length))

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.copy(self.sentences[:self.num_batches * self.batch_size])
        ydata = np.copy(self.next_chars[:self.num_batches * self.batch_size])
        self.x_batches = np.split(xdata, self.num_batches)
        self.y_batches = np.split(ydata, self.num_batches)