import json, sys, os, pickle
import argparse
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

import time
import math

t0 = time.time()

def to_inputs(sentence, vocab, max_len):
    chars = [vocab.by_word(ch, oov_word="<unk>") for ch in sentence]
    chars.append(vocab.by_word("<end>"))
    for i in xrange(max_len - len(sentence) - 1):
        chars.append(vocab.by_word("<pad>"))
    return numpy.asarray(chars)

session = "1_bigcorpus_z128_alpha0.2_max64"

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

n = 33 # an odd number, so there's one in the center!

encoder = model.layers[0].branches[0]
sampler = encoder[-1]

start_words = numpy.ones(n) * start_word

start_words = theano.shared(start_words.astype('int32'))
#sampled = theano.shared(sampled.astype(theano.config.floatX))

decoder_from_z = model.layers[1].branches[0]
x = T.fmatrix('x')
from_z = decoder_from_z(x) #sampled.astype(theano.config.floatX))

layers = model.layers[-3:]
layers[0] = LNLSTMStep(layers[0])
step = Sequential(layers)
embed = model.layers[1].branches[1].layers[-1]

#onehot = OneHot(n_classes + 3) # <unk>, <pad>, <end> I think?

words = start_words
generated = []

#print(from_z)
#print(words)

for i in xrange(max_len):
    #print(onehot(words))
    ins = T.concatenate([from_z[i], embed(words)], axis=1)
    pred = step(ins)
    words = T.argmax(pred, axis=1)
    generated.append(words[None, :])

generated = T.concatenate(generated, axis=0)
f = theano.function([x], outputs=generated)

print("Compiled!")
print("Ready for serving...")

def encode(s1, n):
    #s1 = "hello"
    s1 = to_inputs(s1, vocab, max_len)
    #encoder = model.layers[0].branches[0]
    #sampler = encoder[-1]
    x = T.imatrix()
    z = encoder(x)
    #print(sampler.mu)
    mu = sampler.mu
    f = theano.function([x], mu)
    ins = numpy.zeros((max_len, 1))
    ins[:, 0] = s1
    z = f(ins.astype('int32'))
    sampled = numpy.repeat(z[0][None, :], n, axis=0)
    return sampled

def explore_dimension(s1, dim, n):
    #s1 = "<unk> caller to a local radio station said cocaine"
    #s2 = "giving up some of its gains as the dollar recovered"
    s1 = to_inputs(s1, vocab, max_len)
    encoder = model.layers[0].branches[0]
    sampler = encoder[-1]
    assert isinstance(sampler, Sampler)
    ins = numpy.zeros((max_len, 1))
    ins[:, 0] = s1
    x = T.imatrix()
    z = encoder(x)
    mu = sampler.mu
    f = theano.function([x], mu)
    z = f(ins.astype('int32'))
    s1_z = z[0]
    print(numpy.amax(s1_z))
    #n = 15
    steps = numpy.linspace(0, 1, n)[:, None]
    s1_z = numpy.repeat(s1_z[None, :], n, axis=0) # make a block of z coords
    #steps = numpy.ones(s1_z.shape)
    z_min = 0
    z_max = 4

    dimension_change = numpy.squeeze(numpy.linspace(z_min, z_max, n)[:, None])

    s1_z[:, dim] = s1_z[:, dim] + dimension_change

    return s1_z

def interpolate(s1, s2, n):
    #s1 = "<unk> caller to a local radio station said cocaine"
    #s2 = "giving up some of its gains as the dollar recovered"
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
    #n = 15
    s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
    s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
    steps = numpy.linspace(0, 1, n)[:, None]
    sampled = s1_z * (1 - steps) + s2_z * steps
    return sampled

def jitter(s1, n):
    t1 = time.time()

    s1 = to_inputs(s1, vocab, max_len)
    encoder = model.layers[0].branches[0]
    sampler = encoder[-1]
    assert isinstance(sampler, Sampler)
    ins = numpy.zeros((max_len, 1))
    ins[:, 0] = s1
    x = T.imatrix()
    z = encoder(x)

    mu = sampler.mu
    f = theano.function([x], mu)
    z = f(ins.astype('int32'))

    s1_z = z[0]
    #n = 15
    num_dims = len(s1_z) # BEFORE we do the repeat

    s1_z = numpy.repeat(s1_z[None, :], n, axis=0)


    for i in xrange(n):
        perturb = numpy.random.rand(num_dims) / 1.5
        s1_z[i, :] = s1_z[i, :] + perturb

    t2 = time.time()
    print("whole jitter thing took %f secs" % (t2-t1) )

    return s1_z

def get_z(s1):
    t1 = time.time()

    s1 = to_inputs(s1, vocab, max_len)
    encoder = model.layers[0].branches[0]
    sampler = encoder[-1]
    assert isinstance(sampler, Sampler)
    ins = numpy.zeros((max_len, 1))
    ins[:, 0] = s1
    x = T.imatrix()
    z = encoder(x)

    mu = sampler.mu
    f = theano.function([x], mu)
    z = f(ins.astype('int32'))

    s1_z = z[0]

    t2 = time.time()
    print("get_z thing took %f secs" % (t2-t1) )

    return s1_z.tolist()

def render_results(w):
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
        print r.strip()

        results.append(r)

    return results

def serve(s1, s2):
    w = f(interpolate(s1,s2,n).astype(theano.config.floatX))
    return render_results(w)

def serve_jitter(s1):
    t1 = time.time()
    w = f(jitter(s1, n).astype(theano.config.floatX))
    t2 = time.time()
    print("serve_jitter f took %f secs" % (t2-t1) )
    t1 = time.time()
    res = render_results(w)
    t2 = time.time()
    print("print took %f secs" % (t2-t1) )
    return res

def serve_mod(z, dim, magnitude):
    t1 = time.time()

    s1_z = numpy.array(z).astype(theano.config.floatX)
    s1_z = numpy.repeat(s1_z[None, :], n, axis=0)

    half_n_floor = int(math.floor(n/2))
    mod_eps = 0.1
    neg_steps = numpy.linspace(-magnitude, -mod_eps, half_n_floor)
    pos_steps = numpy.linspace(mod_eps, magnitude, half_n_floor)

    steps = numpy.zeros(n)
    # this is all a bit awkward...
    steps[0:half_n_floor] = neg_steps
    steps[half_n_floor+1] = 0.0
    steps[-half_n_floor:] = pos_steps

    offset = 8
    steps = numpy.repeat(steps[None, :], offset, axis=0)
    steps = numpy.rollaxis(steps, 1) # !!!???
    print(steps.shape)

    s1_z[:, (dim*offset):(dim*offset)+offset] += steps

    w = f(s1_z.astype(theano.config.floatX))

    t2 = time.time()
    print("serve_mod f took %f secs" % (t2-t1) )

    t1 = time.time()
    res = render_results(w)

    t2 = time.time()
    print("print took %f secs" % (t2-t1) )

    return s1_z.tolist(), res

def serve_get_z(s1):
    t1 = time.time()
    z = get_z(s1) # a list
    t2 = time.time()
    print("serve_get_z f took %f secs" % (t2-t1) )
    return z

t1 = time.time()
print("Model startup took %i seconds" % (t1-t0))

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/textserve', methods=['GET'])
def textserve():
    s1 = request.args.get('s1')
    s2 = request.args.get('s2')
    print(s1)
    print(s2)
    results = serve(s1, s2)
    return ''.join(results)

@app.route('/textserve_jitter', methods=['GET'])
def textserve_jitter():
    s1 = request.args.get('s1')
    print(s1)
    results = serve_jitter(s1)
    return ''.join(results)

@app.route('/textserve_get_z', methods=['POST'])
def textserve_get_z():
    json = request.get_json()
    z = serve_get_z(json["s1"])
    return jsonify({"z": z})

@app.route('/textserve_mod', methods=['POST'])
def textserve_mod():
    json = request.get_json()
    z, text = serve_mod(json["z"], json["dim"], json["magnitude"])
    return jsonify({"z": z, "text": text})

"""
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
"""

