import pickle
import sys
from nn.utils import Vocabulary

session = sys.argv[1]

vocab_file = open("session/%s/vocab.pkl" % session)
vocab = Vocabulary()

vocab = pickle.load(vocab_file)
vocab_file.close

print(vocab.word_to_index)
