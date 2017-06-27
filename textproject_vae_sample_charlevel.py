import pickle
import argparse
import os
from scipy.stats import norm
import numpy
import theano
import theano.tensor as T
from databases.textproject_reconstruction_database import TextProjectReconstructionDatabase

from nn.containers import Sequential
from nn.rnns import LNLSTM
from nn.layers import OneHot
from nn.utils import Vocabulary
import nn.utils

from lm_vae import Sampler
from lm_vae_sample import LNLSTMStep
from textproject_vae_charlevel import make_model


def main(session):
    vocab = Vocabulary()

    if os.path.exists("session/%s/vocab.pkl" % session):
        with open("session/%s/vocab.pkl" % session) as vocab_file:
           vocab = pickle.load(vocab_file)
           print("Loaded vocab with %i chars:" % len(vocab))
           print(vocab.index_to_word)
    else:
        print("Using default 256-char vocab")
        # old-school
        vocab.add("<pad>")
        vocab.add("<unk>")
        vocab.add("<end>")
        for i in xrange(256):
            ch = chr(i)
            vocab.add(ch)

    n_classes = len(vocab)

    hyper_params = nn.utils.read_json("session/%s/hyper_params.json" % session)

    z = hyper_params["z"]
    max_len = hyper_params["max_len"]
    p = hyper_params["p"]
    lstm_size = hyper_params["lstm_size"]
    alpha = hyper_params["alpha"]
    dataset = hyper_params["dataset"]

    print("Loading session %s" % session)
    print("Trained using dataset %s" % session)
    print("z: %s, max_len: %s, p: %s, lstm_size: %s, alpha: %s" % (z, max_len, p, lstm_size, alpha))

    model = make_model(z, max_len, p, n_classes, lstm_size, alpha)
    model.load("session/%s/model.flt" % session)
    model.set_phase(train=False)

    start_word = n_classes

    mode = 'custom'
    #mode = 'interpolate'

    if mode == 'vary':
        n = 7
        sampled = numpy.random.normal(0, 1, (1, z))
        sampled = numpy.repeat(sampled, n * z, axis=0)
        for dim in xrange(z):
            eps = 0.01
            x = numpy.linspace(eps, 1 - eps, num=n)
            x = norm.ppf(x)
            sampled[dim*n:(dim+1)*n, dim] = x
        n *= z
    elif mode == 'interpolatereal':
        valid_db = TextProjectReconstructionDatabase(dataset=dataset, phase="valid", batch_size=50, batches_per_epoch=100, max_len=max_len)
        s1 = numpy.random.randint(0, len(valid_db.sentences))
        s2 = numpy.random.randint(0, len(valid_db.sentences))
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((max_len, 2))
        ins[:, 0] = valid_db.to_inputs(valid_db.sentences[s1])
        ins[:, 1] = valid_db.to_inputs(valid_db.sentences[s2])
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        n = 7
        s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
        s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
        steps = numpy.linspace(0, 1, n)[:, None]
        sampled = s1_z * (1 - steps) + s2_z * steps
    elif mode == 'arithm':
        valid_db = TextProjectReconstructionDatabase("valid", 50, batches_per_epoch=100, max_len=max_len)
        s1 = numpy.random.randint(0, len(valid_db.sentences))
        s2 = numpy.random.randint(0, len(valid_db.sentences))
        s3 = numpy.random.randint(0, len(valid_db.sentences))
        print valid_db.sentences[s1]
        print valid_db.sentences[s2]
        print valid_db.sentences[s3]
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((max_len, 3))
        ins[:, 0] = valid_db.to_inputs(valid_db.sentences[s1])
        ins[:, 1] = valid_db.to_inputs(valid_db.sentences[s2])
        ins[:, 2] = valid_db.to_inputs(valid_db.sentences[s3])
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        s3_z = z[1]
        n = 1
        sampled = s1_z - s2_z + s3_z
        sampled = sampled[None, :]
    elif mode == 'interpolate':
        z = numpy.random.normal(0, 1, (2, z))
        s1_z = z[0]
        s2_z = z[1]
        n = 7
        s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
        s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
        steps = numpy.linspace(0, 1, n)[:, None]
        sampled = s1_z * (1 - steps) + s2_z * steps
    elif mode == 'custom':
        s1 = "The rocket rose over the planet's surface."
        s2 = "I love you!"
        s1 = to_inputs(s1, vocab, max_len)
        s2 = to_inputs(s2, vocab, max_len)
        encoder = model.layers[0].branches[0]
        sampler = encoder[-1]
        assert isinstance(sampler, Sampler)
        ins = numpy.zeros((max_len, 2))
        ins[:, 0] = s1
        ins[:, 1] = s2
        x = T.imatrix()
        z = encoder(x)
        mu = sampler.mu
        f = theano.function([x], mu)
        z = f(ins.astype('int32'))
        s1_z = z[0]
        s2_z = z[1]
        n = 15
        s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
        s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
        steps = numpy.linspace(0, 1, n)[:, None]
        sampled = s1_z * (1 - steps) + s2_z * steps
    else:
        n = 100
        sampled = numpy.random.normal(0, 1, (n, z))

    start_words = numpy.ones(n) * start_word
    start_words = theano.shared(start_words.astype('int32'))
    sampled = theano.shared(sampled.astype(theano.config.floatX))

    decoder_from_z = model.layers[1].branches[0]
    from_z = decoder_from_z(sampled)

    layers = model.layers[-3:]
    layers[0] = LNLSTMStep(layers[0])
    step = Sequential(layers)
    embed = model.layers[1].branches[1].layers[-1]

    words = start_words
    generated = []
    for i in xrange(max_len):
        ins = T.concatenate([from_z[i], embed(words)], axis=1)
        pred = step(ins)
        words = T.argmax(pred, axis=1)
        generated.append(words[None, :])

    generated = T.concatenate(generated, axis=0)
    import time
    t = time.time()
    print "compiling...",
    f = theano.function([], outputs=generated)
    print "done, took %f secs" % (time.time() - t)
    w = f()

    results = []

    pad = vocab.by_word("<pad>")
    end = vocab.by_word("<end>")
    for i in xrange(w.shape[1]):
        s = []
        for idx in w[:, i]:
            if idx == end:
                break
            if idx == pad:
                break
            s.append(vocab.by_index(idx))
        r = ''.join(s)
        if mode == "vary":
            if i % n == 0:
                print "dimension %d" % (i / n)
        print r.strip()
        results.append(r)

def to_inputs(tweet, vocab, max_len):
    chars = [vocab.by_word(ch, oov_word='<unk>') for ch in tweet]
    chars.append(vocab.by_word('<end>'))
    for i in xrange(max_len - len(tweet) - 1):
        chars.append(vocab.by_word('<pad>'))
    return numpy.asarray(chars)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-z', default=100, type=int)
    #parser.add_argument('-p', default=0.0, type=float)
    #parser.add_argument('-alpha', default=0.2, type=float)
    #parser.add_argument('-sample_size', default=128, type=int)
    #parser.add_argument('-lstm_size', default=1000, type=int)
    parser.add_argument('-session', type=str)
    args = parser.parse_args()
    main(**vars(args))
