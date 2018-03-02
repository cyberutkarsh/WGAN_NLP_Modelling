"""Util file to read and process data."""
import numpy as np
import torch
from torch.autograd import Variable
import operator
import string
from nltk.grammar import CFG
from nltk.parse.earleychart import EarleyChartParser
from nltk.parse.viterbi import ViterbiParser
from nltk.parse.generate import generate
from nltk.translate import bleu_score
import pickle
import os


def minibatch_to_1hot(minibatch, vocab_size):
    """Convert minibatch into sequence of 1hot vectors."""
    minibatch_1hot = np.zeros(
        (minibatch.shape[0], minibatch.shape[1], vocab_size)
    ).astype(np.float32)

    for ind, sentence in enumerate(minibatch):
        for ind2, word in enumerate(sentence):
            minibatch_1hot[ind][ind2][word] = 1.

    return Variable(torch.from_numpy(minibatch_1hot)).cuda()


def get_noise_vector(vocab_size, correct_ind, start_noise):
    """Construct a noisy 1-hot vector."""
    correct_mass = np.random.uniform(start_noise, 1)
    remainder = 1. - correct_mass
    while True:
        num_parts = np.random.randint(2, vocab_size - 1)
        indices = np.random.choice(range(vocab_size), num_parts)
        if correct_ind not in indices:
            break

    index_ratio = {ind: np.random.random() for ind in indices}
    summ = np.sum(index_ratio.values())
    for ind, mass in index_ratio.items():
        index_ratio[ind] = mass / summ

    index_mass = {ind: index_ratio[ind] * remainder for ind in indices}
    index_mass[correct_ind] = correct_mass
    return index_mass


def minibatch_to_1hotnoise(minibatch, vocab_size):
    """Convert minibatch into sequence of noise injected 1hot vectors."""
    minibatch_1hot = np.zeros(
        (minibatch.shape[0], minibatch.shape[1], vocab_size)
    ).astype(np.float32)

    for ind, sentence in enumerate(minibatch):
        for ind2, word in enumerate(sentence):
            # __TOD__ scale noise vector to be directly proportional to ind2
            noise_vector = get_noise_vector(vocab_size, word, start_noise=0.5)
            for index, mass in noise_vector.items():
                minibatch_1hot[ind][ind2][index] = mass

    return Variable(torch.from_numpy(minibatch_1hot)).cuda()


def minibatch_to_1hotnoise_fast(minibatch, vocab_size):
    """Convert minibatch into sequence of 1hot vectors."""
    minibatch_1hot = np.zeros(
        (minibatch.shape[0], minibatch.shape[1], vocab_size)
    ).astype(np.float32)

    for ind, sentence in enumerate(minibatch):
        for ind2, word in enumerate(sentence):
            minibatch_1hot[ind][ind2][word] = 1.

    minibatch_1hot += np.random.normal(loc=0, scale=0.05, size=minibatch_1hot.shape)

    return Variable(torch.from_numpy(minibatch_1hot)).cuda()


def softmax_3d(x):
    """3D softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s


def minibatch_to_1hotnoise_fast_softmax(minibatch, vocab_size):
    """Convert minibatch into sequence of 1hot vectors."""
    minibatch_1hot = np.zeros(
        (minibatch.shape[0], minibatch.shape[1], vocab_size)
    ).astype(np.float32)

    for ind, sentence in enumerate(minibatch):
        for ind2, word in enumerate(sentence):
            minibatch_1hot[ind][ind2][word] = 1.

    minibatch_1hot += np.random.normal(loc=0, scale=0.01, size=minibatch_1hot.shape)
    minibatch_1hot = softmax_3d(minibatch_1hot)

    return Variable(torch.from_numpy(minibatch_1hot)).cuda()


class DataIterator:
    """Data iterator."""

    def __init__(self, data_path, add_start_end=False, trim_vocab=True, vocab_size=30000):
        """Initialize params."""
        self.data_path = data_path
        self.sentences = None
        self.sentence_buckets = None
        self.word2id = None
        self.id2word = None
        self.add_start_end = add_start_end
        self.trim_vocab = trim_vocab
        self.vocab_size = vocab_size

    def read_data(self):
        """Read data."""
        self.sentences = [
            line.strip().lower().split() for line in open(self.data_path, 'r')
        ]
        self.sentence_buckets = {}
        for sentence in self.sentences:
            if self.add_start_end:
                sentence = ['<s>'] + sentence + ['</s>']
            length = len(sentence)
            if length not in self.sentence_buckets:
                self.sentence_buckets[length] = [sentence]
            else:
                self.sentence_buckets[length].append(sentence)

    def _trim_vocab(self, vocab, vocab_size):

        # Discard start, end, pad and unk tokens if already present
        if '<s>' in vocab:
            del vocab['<s>']
        if '<pad>' in vocab:
            del vocab['<pad>']
        if '</s>' in vocab:
            del vocab['</s>']
        if '<unk>' in vocab:
            del vocab['<unk>']

        word2id = {
            '<s>': 0,
            '<pad>': 1,
            '</s>': 2,
            '<unk>': 3,
        }

        id2word = {
            0: '<s>',
            1: '<pad>',
            2: '</s>',
            3: '<unk>',
        }

        sorted_word2id = sorted(
            vocab.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]

        for ind, word in enumerate(sorted_words):
            word2id[word] = ind + 4

        for ind, word in enumerate(sorted_words):
            id2word[ind + 4] = word

        return word2id, id2word

    def compute_vocab(self):
        """Compute vocabulary."""
        self.vocab = {}
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = 1
                else:
                    self.vocab[word] += 1

        if self.trim_vocab:
            self.word2id, self.id2word = self._trim_vocab(self.vocab, self.vocab_size)
        else:
            self.word2id = {word: ind for ind, word in enumerate(self.vocab.keys())}
            self.id2word = {ind: word for ind, word in enumerate(self.vocab.keys())}

    def compute_bigrams(self):
        """Compute all bigrams in dataset."""
        self.bigrams = {}
        for sentence in self.sentences:
            sentence = ['<s>'] + sentence + ['</s>']
            for ind in xrange(len(sentence) - 1):
                bigram = tuple(sentence[ind: ind + 2])
                if bigram not in self.bigrams:
                    self.bigrams[bigram] = 1
                else:
                    self.bigrams[bigram] += 1

    def compute_trigrams(self):
        """Compute all trigrams in dataset."""
        self.trigrams = {}
        for sentence in self.sentences:
            sentence = ['<s>'] + sentence + ['</s>']
            for ind in xrange(len(sentence) - 2):
                trigram = tuple(sentence[ind: ind + 3])
                if trigram not in self.trigrams:
                    self.trigrams[trigram] = 1
                else:
                    self.trigrams[trigram] += 1

    def get_data_by_length(self, length, num_items):
        """Get data based on length."""
        if length not in self.sentence_buckets:
            return []
        num_items = num_items if num_items < len(self.sentence_buckets[length]) \
            else len(self.sentence_buckets[length])
        random_sentence_inds = np.random.choice(
            range(len(self.sentence_buckets[length])),
            num_items,
            replace=False
        )
        random_sentences = [self.sentence_buckets[length][ind] for ind in random_sentence_inds]
        return random_sentences

    def get_random_minibatch(self, num_items, index):
        return self.sentences[index: index + num_items]


class CharDataIterator:
    """Data iterator."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self.sentences = None
        self.sentence_buckets = None
        self.word2id = None
        self.id2word = None

    def _replace_punctuations(self, sentence):
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        sentence = sentence.translate(replace_punctuation)
        return sentence

    def read_data(self):
        """Read data."""
        self.sentences = [
            self._replace_punctuations(line.strip().lower()) for line in open(self.data_path, 'r')
        ]
        self.sentence_buckets = {}
        for sentence in self.sentences:
            length = len(sentence)
            if length not in self.sentence_buckets:
                self.sentence_buckets[length] = [sentence]
            else:
                self.sentence_buckets[length].append(sentence)

    def compute_vocab(self):
        """Compute vocabulary."""
        self.word2id = {}
        self.id2word = {}
        ind = 0
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.word2id:
                    self.word2id[word] = ind
                    self.id2word[ind] = word
                    ind += 1

        '''
        self.word2id['<pad>'] = ind
        self.id2word[ind] = '<pad>'
        if '<s>' not in self.word2id:
            self.word2id['<s>'] = ind
            self.id2word[ind] = '<s>'
        if '</s>' not in self.word2id:
            self.word2id['</s>'] = ind + 1
            self.id2word[ind + 1] = '</s>'
        '''

    def get_data_by_length(self, length, num_items):
        """Get data based on length."""
        if length not in self.sentence_buckets:
            return []
        num_items = num_items if num_items < len(self.sentence_buckets[length]) \
            else len(self.sentence_buckets[length])
        random_sentence_inds = np.random.choice(
            range(len(self.sentence_buckets[length])),
            num_items,
            replace=False
        )
        random_sentences = [self.sentence_buckets[length][ind] for ind in random_sentence_inds]
        return random_sentences

    def get_random_minibatch(self, num_items, index):
        return self.sentences[index: index + num_items]


class TruncCharDataIterator:
    """Data iterator."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self.sentences = None
        self.sentence_buckets = None
        self.word2id = None
        self.id2word = None

    def _replace_punctuations(self, sentence):
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        sentence = sentence.translate(replace_punctuation)
        return sentence

    def read_data(self):
        """Read data."""
        self.sentences = [
            self._replace_punctuations(line.strip().lower()) for line in open(self.data_path, 'r')
        ]

    def compute_vocab(self):
        """Compute vocabulary."""
        self.word2id = {}
        self.id2word = {}
        ind = 0
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.word2id:
                    self.word2id[word] = ind
                    self.id2word[ind] = word
                    ind += 1

        '''
        self.word2id['<pad>'] = ind
        self.id2word[ind] = '<pad>'
        if '<s>' not in self.word2id:
            self.word2id['<s>'] = ind
            self.id2word[ind] = '<s>'
        if '</s>' not in self.word2id:
            self.word2id['</s>'] = ind + 1
            self.id2word[ind + 1] = '</s>'
        '''

    def get_data_by_length(self, length, num_items):
        """Get data based on length."""
        if length not in self.sentence_buckets:
            return []
        num_items = num_items if num_items < len(self.sentence_buckets[length]) \
            else len(self.sentence_buckets[length])
        random_sentence_inds = np.random.choice(
            range(len(self.sentence_buckets[length])),
            num_items,
            replace=False
        )
        random_sentences = [self.sentence_buckets[length][ind] for ind in random_sentence_inds]
        return random_sentences

    def get_random_minibatch(self, num_items, index, maxlen=10):
        sentences = self.sentences[index: index + num_items]
        return [x[:maxlen] for x in sentences]


class HolyGrammar:
    """Holy Grammar Class."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self._get_grammar()
        self.parser = EarleyChartParser(self.holy_grammar)
        self._get_trainining_data()

    def _get_grammar(self):
        holygrail_cfg = [line.strip() for line in open(os.path.join(self.data_path, 'holygrail.cfg'), 'r')]
        holygrail_cfg = [line.split() for line in holygrail_cfg if not line.startswith('#') and line != '']
        holygrail_productions = []
        extra_list = ['!', '.', '?', '-', ';', ':', '5', ',']
        for production in holygrail_cfg:
            if len(production) >= 4:
                holygrail_productions.append(production[1] + ' -> %s ' % (' '.join(production[2:])))
            elif len(production) == 3 and (production[-1][0].islower() or production[-1][0] in extra_list):
                holygrail_productions.append(production[1] + ' -> ' + "'" + production[2] + "'")
            elif len(production) == 3 and production[-1][0].isupper():
                holygrail_productions.append(production[1] + ' -> ' + production[2])
            else:
                print ('Found something unexpected')
                print (production)
        self.holy_grammar = CFG.fromstring('\n'.join(holygrail_productions))

    def _get_trainining_data(self):
        self.train_lines = [line.strip().split() for line in open(os.path.join(self.data_path, 'holy_grammar_sentences_diverse.train.txt'), 'r')]

    def is_in_training(self, sentence):
        sentence = sentence.split()
        for item in self.train_lines:
            if item == sentence:
                return True

        return False

    def is_valid_parse(self, sentence):
        return len(list(self.parser.parse(sentence.split()))) != 0


class HolyGrammar11:
    """Holy Grammar Class."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self._get_grammar()
        self.parser = EarleyChartParser(self.holy_grammar)
        self._get_trainining_data()

    def _get_grammar(self):
        holygrail_cfg = [line.strip() for line in open(os.path.join(self.data_path, 'holygrail.cfg'), 'r')]
        holygrail_cfg = [line.split() for line in holygrail_cfg if not line.startswith('#') and line != '']
        holygrail_productions = []
        extra_list = ['!', '.', '?', '-', ';', ':', '5', ',']
        for production in holygrail_cfg:
            if len(production) >= 4:
                holygrail_productions.append(production[1] + ' -> %s ' % (' '.join(production[2:])))
            elif len(production) == 3 and (production[-1][0].islower() or production[-1][0] in extra_list):
                holygrail_productions.append(production[1] + ' -> ' + "'" + production[2] + "'")
            elif len(production) == 3 and production[-1][0].isupper():
                holygrail_productions.append(production[1] + ' -> ' + production[2])
            else:
                print ('Found something unexpected')
                print (production)
        self.holy_grammar = CFG.fromstring('\n'.join(holygrail_productions))

    def _get_trainining_data(self):
        self.train_lines = [line.strip().split() for line in open(os.path.join(self.data_path, 'holy_grammar_sentences_diverse_11.train.txt'), 'r')]

    def is_in_training(self, sentence):
        sentence = sentence.split()
        for item in self.train_lines:
            if item == sentence:
                return True

        return False

    def is_valid_parse(self, sentence):
        return len(list(self.parser.parse(sentence.split()))) != 0


class HolyGrammar8:
    """Holy Grammar Class."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self._get_grammar()
        self.parser = EarleyChartParser(self.holy_grammar)
        self._get_trainining_data()

    def _get_grammar(self):
        holygrail_cfg = [line.strip() for line in open(os.path.join(self.data_path, 'holygrail.cfg'), 'r')]
        holygrail_cfg = [line.split() for line in holygrail_cfg if not line.startswith('#') and line != '']
        holygrail_productions = []
        extra_list = ['!', '.', '?', '-', ';', ':', '5', ',']
        for production in holygrail_cfg:
            if len(production) >= 4:
                holygrail_productions.append(production[1] + ' -> %s ' % (' '.join(production[2:])))
            elif len(production) == 3 and (production[-1][0].islower() or production[-1][0] in extra_list):
                holygrail_productions.append(production[1] + ' -> ' + "'" + production[2] + "'")
            elif len(production) == 3 and production[-1][0].isupper():
                holygrail_productions.append(production[1] + ' -> ' + production[2])
            else:
                print ('Found something unexpected')
                print (production)
        self.holy_grammar = CFG.fromstring('\n'.join(holygrail_productions))

    def _get_trainining_data(self):
        self.train_lines = [line.strip().split() for line in open(os.path.join(self.data_path, 'holy_grammar_sentences_diverse_8.train.txt'), 'r')]

    def is_in_training(self, sentence):
        sentence = sentence.split()
        for item in self.train_lines:
            if item == sentence:
                return True

        return False

    def is_valid_parse(self, sentence):
        return len(list(self.parser.parse(sentence.split()))) != 0


class PTBGrammar:
    """PTB Grammar Class."""

    def __init__(self, data_path):
        """Initialize params."""
        self.data_path = data_path
        self._get_grammar()
        self.parser = ViterbiParser(self.grammar)

    def _get_grammar(self):
        self.grammar = pickle.load(open(os.path.join(self.data_path, 'ptb_indcuded.pcfg.pkl'), 'r'))

    def likelihood(self, sentence):
        try:
            return list(self.parser.parse(sentence.split()))[0].prob()
        except (ValueError, IndexError):
            print ('Vocab error :/ ')
            return None

def test_bleu_5(hyp, n=2):
    hyp = [line.strip().split() for line in open(hyp, 'r')]
    gold = [line.strip().split() for line in open('/data/lisatmp4/subramas/datasets/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtest_5', 'r')]
    all_gold = [gold for i in range(len(hyp))]
    weights = (1,) if n == 2 else (0.5, 0.5)
    return bleu_score.corpus_bleu(all_gold, hyp, weights=weights)


def test_bleu_7(hyp, n=2):
    hyp = [line.strip().split() for line in open(hyp, 'r')]
    gold = [line.strip().split() for line in open('/data/lisatmp4/subramas/datasets/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtest_7', 'r')]
    all_gold = [gold for i in range(len(hyp))]
    weights = (1,) if n == 2 else (0.5, 0.5)
    return bleu_score.corpus_bleu(all_gold, hyp, weights=weights)


def valid_bleu_5(hyp, n=2):
    hyp = [line.strip().split() for line in open(hyp, 'r')]
    gold = [line.strip().split() for line in open('/data/lisatmp4/subramas/datasets/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qvalid_5', 'r')]
    all_gold = [gold for i in range(len(hyp))]
    weights = (1,) if n == 2 else (0.5, 0.5)
    return bleu_score.corpus_bleu(all_gold, hyp, weights=weights)


def valid_bleu_7(hyp, n=2):
    hyp = [line.strip().split() for line in open(hyp, 'r')]
    gold = [line.strip().split() for line in open('/data/lisatmp4/subramas/datasets/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qvalid_7', 'r')]
    all_gold = [gold for i in range(len(hyp))]
    weights = (1,) if n == 2 else (0.5, 0.5)
    return bleu_score.corpus_bleu(all_gold, hyp, weights=weights)


class ConditionalDataIterator:
    """Data iterator."""

    def __init__(self, data_path, remove_numbers=True):
        """Initialize params."""
        self.data_path = data_path
        self.sentences = None
        self.sentence_buckets = None
        self.word2id = None
        self.id2word = None
        self.remove_numbers = remove_numbers

    def read_data(self):
        """Read data."""
        sentences = [
            line.strip().split('\t') for line in open(self.data_path, 'r')
        ]
        sentence_content = [x[0].split() for x in sentences]
        sentence_labels = [x[1] for x in sentences]
        self.sentence_buckets = {}
        for sentence, label in zip(sentence_content, sentence_labels):
            length = len(sentence)
            if length not in self.sentence_buckets:
                self.sentence_buckets[length] = [(sentence, label)]
            else:
                self.sentence_buckets[length].append((sentence, label))

        self.sentences = sentence_content

    def compute_vocab(self):
        """Compute vocabulary."""
        self.word2id = {}
        self.id2word = {}
        ind = 0
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.word2id:
                    self.word2id[word] = ind
                    self.id2word[ind] = word
                    ind += 1

        self.word2id['<pad>'] = ind
        self.id2word[ind] = '<pad>'
        if '<s>' not in self.word2id:
            self.word2id['<s>'] = ind
            self.id2word[ind] = '<s>'
        if '</s>' not in self.word2id:
            self.word2id['</s>'] = ind + 1
            self.id2word[ind + 1] = '</s>'

    def get_data_by_length(self, length, num_items):
        """Get data based on length."""
        if length not in self.sentence_buckets:
            return []
        num_items = num_items if num_items < len(self.sentence_buckets[length]) \
            else len(self.sentence_buckets[length])
        random_sentence_inds = np.random.choice(
            range(len(self.sentence_buckets[length])),
            num_items,
            replace=False
        )
        random_sentences = [self.sentence_buckets[length][ind] for ind in random_sentence_inds]
        return random_sentences


if __name__ == '__main__':
    data_iterator = ConditionalDataIterator(data_path='/data/lisatmp4/subramas/datasets/cmu_hw_data/cmu-mthomework.conditonal.train.en.unk')
    data_iterator.read_data()
    data_iterator.compute_vocab()
    print ('Found %d sentences' % (len(data_iterator.sentences)))
    print (data_iterator.get_data_by_length(8, 32))
