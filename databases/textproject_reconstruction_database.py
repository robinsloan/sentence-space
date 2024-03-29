import numpy
import theano
import theano.tensor as T
import os
import time
from nn.utils import Vocabulary


class TextProjectReconstructionDatabase(object):

    def __init__(self, dataset, phase, batch_size, max_len=140, pad=True, sp_model=None):
        self.phase = phase
        self.batch_size = batch_size
        self.max_len = max_len
        self.sp_model = sp_model
       
        self.vocab = Vocabulary()

        self.using_sp = (self.sp_model != None) and (len(self.sp_model) > 0)
        if self.using_sp:
            print("Using sentencepiece")
            import sentencepiece as spm # https://github.com/google/sentencepiece

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self.sp_model)
            sp_model_size = self.sp.GetPieceSize()
            print("Loaded SP model with", sp_model_size, "tokens")
            self.vocab.add('<pad>')
            self.vocab.add('<unk>')
            self.vocab.add('<end>')

            for i in xrange(sp_model_size):
                self.vocab.add(self.sp.IdToPiece(i))

        else:
            print("Using default fixed vocab")
            self.vocab.add('<pad>')
            self.vocab.add('<unk>')
            self.vocab.add('<end>')
            for i in xrange(32, 128):
                ch = chr(i)
                self.vocab.add(ch)

        self.n_classes = len(self.vocab)
        self.pad = pad

        self.sentences = []

        # First, check full path; if that doesn't work, make a guess that it's in data/
        if os.path.exists(dataset):
            dataset_path = dataset
        elif os.path.exists("data/%s" % dataset):
            dataset_path = "data/%s" % dataset
        else:
            raise Exception("Can't find any dataset named %s!" % dataset)

        with open(dataset_path) as f:
            while True:
                s = f.readline()
                if s == "":
                    break
                if self.using_sp:
                    chars = self.sp.EncodeAsIds(s)
                    if len(chars) <= max_len - 1:
                        self.sentences.append(s)
                elif len(s) <= max_len - 1:
                    self.sentences.append(s)
                #if len(self.sentences) >= 1000000:
                    #break


        self.shuffle_sentences()

        valid_size = int(len(self.sentences) * 0.1)

        if self.phase == 'train':
            self.sentences = self.sentences[valid_size:]
        else:
            self.sentences = self.sentences[:valid_size]

        print "%s data: %d sentences, max %d chars" % (phase, len(self.sentences), max_len)

        self.batches_per_epoch = int(len(self.sentences) / batch_size)

        # per the original textvae code, let's just keep this lean
        if self.phase == 'valid':
            print "Reducing valid set to 100 batches"
            self.batches_per_epoch = min(self.batches_per_epoch, 100)

        x = self.make_batch()

        self.shared_x = theano.shared(x)

        self.index = T.iscalar()

    def shuffle_sentences(self):
        # this all might be horribly inefficient but whatever
        t = time.time()
        print("Shuffling %s sentences..." % len(self.sentences))
        numpy.random.shuffle(self.sentences)
        print "...done. Took %s seconds" % round(time.time() - t)

    def to_inputs(self, sentence):
        sentence = sentence.replace("\n","")
        if self.using_sp:
            #print(self.sp.EncodeAsPieces(sentence))
            chars = self.sp.EncodeAsIds(sentence)
        else:
            chars = [self.vocab.by_word(ch, oov_word='<unk>') for ch in sentence]
        chars.append(self.vocab.by_word('<end>'))
        for i in xrange(self.max_len - len(chars)):
            chars.append(self.vocab.by_word('<pad>'))
        return numpy.asarray(chars)

    # The original code drew random samples but didn't keep track of which had already been drawn. This seems not ideal to me so I am rewriting to make minibatches draw samples *without* replacement.
    # EDIT: Now back to doing it the original way, because speed!

    def make_batch(self):
        batch = numpy.zeros((self.max_len, self.batch_size))

        if self.pad:
            for i in xrange(self.batch_size):
                idx = numpy.random.randint(len(self.sentences))
                # here we pop the sentence out of the array entirely
                batch[:, i] = self.to_inputs(self.sentences[idx])
        else:
            idx = numpy.random.randint(len(self.sentences))
            max_len = len(self.sentences[idx])
            target_len = len(self.sentences[idx])
            batch[:, 0] = self.to_inputs(self.sentences[idx])
            i = 1
            while i < self.batch_size:
                idx = numpy.random.randint(len(self.sentences))
                if abs(len(self.sentences[idx]) - target_len) > 3:
                    continue
                batch[:, i] = self.to_inputs(self.sentences[idx])
                max_len = max(max_len, len(self.sentences[idx]) + 1)
                i += 1
            batch = batch[0:max_len]

        return batch.astype('int32')

    def givens(self, x, t):
        return {
            x: self.shared_x[:, self.index * self.batch_size:(self.index+1) * self.batch_size],
        }

    def total_batches(self):
        return self.batches_per_epoch

    def indices(self):
        for i in xrange(self.batches_per_epoch):
            x = self.make_batch()
            self.shared_x.set_value(x)
            yield 0
