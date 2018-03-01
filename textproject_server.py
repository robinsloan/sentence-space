#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, sys, os, pickle, time, math
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

from wordfilter import Wordfilter
wordfilter = Wordfilter()

t1 = time.time()

session = "sp15_trial"

vocab = Vocabulary()

if os.path.exists("session/%s/vocab.pkl" % session):
    with open("session/%s/vocab.pkl" % session) as vocab_file:
       vocab = pickle.load(vocab_file)
       print("Loaded vocab with %i chars:" % len(vocab))
       #print(vocab.index_to_word)
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
sp_model = str(hyper_params["sp_model"]) # I get an error below if i don't cast to string...

if sp_model:
    import sentencepiece as spm # https://github.com/google/sentencepiece
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)

print("Loading session %s" % session)
print("Trained using dataset %s" % session)
print("z: %s, max_len: %s, p: %s, lstm_size: %s, alpha: %s" % (z, max_len, p, lstm_size, alpha))

model = make_model(z, max_len, p, n_classes, lstm_size, alpha)
model.load("session/%s/model.flt" % session)
model.set_phase(train=False)

start_word = n_classes

n = 7 # an odd number, so there's one in the center!

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

t2 = time.time()
print("Startup took %i seconds!" % (t2-t1))
print("Ready for serving.")

def to_inputs(sentence):
    sentence = str(sentence) # ???
    if sp:
        #print(self.sp.EncodeAsPieces(sentence))
        chars = sp.EncodeAsIds(sentence)
    else:
        chars = [vocab.by_word(ch, oov_word='<unk>') for ch in sentence]
    print(chars)

    pad = vocab.by_word("<pad>")
    end = vocab.by_word("<end>")

    s = []
    for char in chars:
        if char == end:
            break
        if char == pad:
            break
        s.append(vocab.by_index(char+2)) # OFFSET BUGGGGG omg omg omg

    readout_str = ""
    for c in s:
        readout_str += c + " "
    end

    print(readout_str)

    chars.append(vocab.by_word('<end>'))
    for i in xrange(max_len - len(chars)):
        chars.append(vocab.by_word('<pad>'))
    return numpy.asarray(chars)

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
            s.append(vocab.by_index(idx+2)) # THIS IS SO GNARLY
        r = ''.join(s)
        print r.strip()

        results.append(r)

    return results

# from: https://github.com/soumith/dcgan.torch/issues/14
def slerp(val, low, high):
    omega = numpy.arccos(numpy.clip(numpy.dot(low/numpy.linalg.norm(low), high/numpy.linalg.norm(high)), -1, 1))
    so = numpy.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return numpy.sin((1.0-val)*omega) / so * low + numpy.sin(val*omega) / so * high

def lerp(val, low, high):
    return (low + (high-low)*val)

def calc_interpolate(s1, s2, num_steps):
    s1 = to_inputs(s1)
    s2 = to_inputs(s2)
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
    #s1_z = numpy.repeat(s1_z[None, :], n, axis=0)
    #s2_z = numpy.repeat(s2_z[None, :], n, axis=0)
    steps = numpy.linspace(0, 1, num_steps)[:, None]
    #sampled = s1_z * (1 - steps) + s2_z * steps

    sampled = numpy.zeros((num_steps, len(s1_z))) # len(s1_z) gives num of z-dims

    # https://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    for index, step in enumerate(steps):
        sampled[index] = lerp(step, s1_z, s2_z)

    return sampled

def calc_jitter(s1, mag):
    s1 = to_inputs(s1)
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

    s1_z_n = numpy.repeat(s1_z[None, :], n, axis=0)

    jitter_scale = float(mag) # 0.2
    # MAKE NEGATIVE TOO
    jitter_vals = (numpy.random.rand(n, len(s1_z)) - 0.5) * 2.0 * jitter_scale
    sampled = s1_z_n + jitter_vals

    return sampled

def calc_get_z(s1):
    s1 = to_inputs(s1)
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

    return s1_z

def serve_interpolate(s1, s2):
    w = f(calc_interpolate(s1, s2, n).astype(theano.config.floatX))
    return render_results(w)

def serve_jitter(s1, mag):
    w = f(calc_jitter(s1, float(mag)).astype(theano.config.floatX))
    return render_results(w)

def serve_get_z(s1):
    s1_z = calc_get_z(s1)
    multi_z = numpy.repeat(s1_z[None, :], n, axis=0)
    w = f(multi_z.astype(theano.config.floatX))
    rendered = render_results(w)[0]
    return s1_z.tolist(), rendered

def screen_results(results):
    screened_results = list(results)
      
    for i in xrange(len(results)):
      if wordfilter.blacklisted(results[i]):
        screened_results[i] = "***"
    
    return screened_results

def process_results(results):
    print("Processing results")
    processed_results = list(results)

    for i in xrange(len(results)):
      processed_results[i] = results[i].decode('utf-8').replace(u"\u2581"," ")[1:]

    return processed_results

#serve_interpolation("I love you.", "She saw the sun rising and knew that it was a bad sign for the future of the planet.")
#quit()

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/gradient', methods=['GET'])
def gradient():
    s1 = request.args.get('s1')[0:max_len]
    s2 = request.args.get('s2')[0:max_len]
    print("Interpolating:")
    print(s1)
    print(s2)
    results = serve_interpolate(s1, s2)
    results = process_results(results)
    results = screen_results(results)
    return jsonify({"results": results})

@app.route('/jitter', methods=['GET'])
def jitter():
    s1 = request.args.get('s1')[0:max_len]
    mag = request.args.get('mag') or 0.1
    print("Jittering:")
    print(s1)
    results = serve_jitter(s1, mag)
    results = process_results(results)
    results = screen_results(results)
    return jsonify({"results": results})

@app.route('/get_z', methods=['GET'])
def get_z():
    #json = request.get_json()
    s1 = request.args.get('s1')
    print(s1)
    z, text = serve_get_z(s1)
    return jsonify({"z": z, "text": text})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-z', default=100, type=int)
    #parser.add_argument('-p', default=0.0, type=float)
    #parser.add_argument('-alpha', default=0.2, type=float)
    #parser.add_argument('-sample_size', default=128, type=int)
    #parser.add_argument('-lstm_size', default=1000, type=int)
    #parser.add_argument('-session', type=str)
    args = parser.parse_args()
    main(**vars(args))


