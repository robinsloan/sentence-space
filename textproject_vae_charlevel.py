import argparse
import pickle

import numpy
import theano
import theano.tensor as T
from databases.lm_reconstruction_database import LmReconstructionDatabase
from databases.textproject_reconstruction_database import TextProjectReconstructionDatabase

from vae import Sampler, Dropword, Store, LMReconstructionModel
from nn.layers1d import LayoutCNNToRNN, LayoutRNNToCNN, Convolution1d, Deconvolution1D, HighwayConvolution1d
from nn.layers import Linear, Embed, Flatten, Reshape, SoftMax, Dropout, OneHot
from nn.activations import ReLU, Tanh, Gated
from nn.models.base_model import BaseModel
from nn.optimizer import Optimizer
from nn.updates import Adam
from nn.rnns import LNLSTM
from nn.containers import Sequential, Parallel
from nn.normalization import BatchNormalization
import nn.utils


def make_model(z, sample_size, dropword_p, n_classes, lstm_size, alpha):
    encoder = [
        OneHot(n_classes),
        LayoutRNNToCNN(),
        Convolution1d(3, 128, n_classes, pad=1, stride=2, causal=False, name="conv1"),
        BatchNormalization(128, name="bn1"),
        ReLU(),
        Convolution1d(3, 256, 128, pad=1, stride=2, causal=False, name="conv2"),
        BatchNormalization(256, name="bn2"),
        ReLU(),
        Convolution1d(3, 512, 256, pad=1, stride=2, causal=False, name="conv3"),
        BatchNormalization(512, name="bn3"),
        ReLU(),
        Convolution1d(3, 512, 512, pad=1, stride=2, causal=False, name="conv4"),
        BatchNormalization(512, name="bn4"),
        ReLU(),
        Convolution1d(3, 512, 512, pad=1, stride=2, causal=False, name="conv5"),
        BatchNormalization(512, name="bn5"),
        ReLU(),
        Flatten(),
        Linear(sample_size / (2**5) * 512, z * 2, name="fc_encode"),
        Sampler(z),
    ]
    decoder_from_z = [
        Linear(z, sample_size / (2**5) * 512, name="fc_decode"),
        ReLU(),
        Reshape((-1, 512, sample_size / (2**5), 1)),
        Deconvolution1D(512, 512, 3, pad=1, stride=2, name="deconv5"),
        BatchNormalization(512, name="deconv_bn5"),
        ReLU(),
        Deconvolution1D(512, 512, 3, pad=1, stride=2, name="deconv4"),
        BatchNormalization(512, name="deconv_bn4"),
        ReLU(),
        Deconvolution1D(512, 256, 3, pad=1, stride=2, name="deconv3"),
        BatchNormalization(256, name="deconv_bn3"),
        ReLU(),
        Deconvolution1D(256, 128, 3, pad=1, stride=2, name="deconv2"),
        BatchNormalization(128, name="deconv_bn2"),
        ReLU(),
        Deconvolution1D(128, 200, 3, pad=1, stride=2, name="deconv1"),
        BatchNormalization(200, name="deconv_bn1"),
        ReLU(),
        LayoutCNNToRNN(),
        Parallel([
            [
                Linear(200, n_classes, name="aux_classifier"),
                SoftMax(),
                Store()
            ],
            []
        ], shared_input=True),
        lambda x: x[1]
    ]

    start_word = n_classes
    dummy_word = n_classes + 1
    decoder_from_words = [
        Dropword(dropword_p, dummy_word=dummy_word),
        lambda x: T.concatenate([T.ones((1, x.shape[1]), dtype='int32') * start_word, x], axis=0),
        lambda x: x[:-1],
        OneHot(n_classes+2),
    ]
    layers = [
        Parallel([
            encoder,
            []
        ], shared_input=True),
        Parallel([
            decoder_from_z,
            decoder_from_words
        ], shared_input=False),
        lambda x: T.concatenate(x, axis=2),
        LNLSTM(200+n_classes+2, lstm_size, name="declstm"),
        Linear(lstm_size, n_classes, name="classifier"),
        SoftMax()
    ]

    model = LMReconstructionModel(layers, aux_loss=True, alpha=alpha)

    return model

def main(z, lr, anneal_start, anneal_end, p, alpha, lstm_size, num_epochs, max_len, batch_size, session, dataset, resume):
    train_db = TextProjectReconstructionDatabase(dataset=dataset, phase="train", batch_size=batch_size, max_len=max_len)
    valid_db = TextProjectReconstructionDatabase(dataset=dataset, phase="valid", batch_size=batch_size, max_len=max_len)

    model = make_model(z, max_len, p, train_db.n_classes, lstm_size, alpha)
    model.anneal_start = float(anneal_start)
    model.anneal_end = float(anneal_end)

    vocab = train_db.vocab

    print(len(vocab))
    print(vocab)

    if resume:
        model.load("session/%s/model.flt" % session)
        print("Resuming session %s" % session)

    #out = nn.utils.forward(model, train_db, out=model.output(model.input))
    #print out.shape
    #return

    print("Total params: %s" % model.total_params)

    # "textproject-charlevel-z_%d-len_%d-p_%.2f-lstmsz_%d-alpha_%.2f" % (z, sample_size, p, lstm_size, alpha)

    opt = Optimizer(model, train_db, valid_db, Adam(lr),
                    name=session, print_info=True, restore=resume)

    with open("%s/vocab.pkl" % opt.opt_folder, "wb") as vocab_file:
        pickle.dump(train_db.vocab, vocab_file)

    nn.utils.save_json("%s/hyper_params.json" % opt.opt_folder, {
        # z, sample_size, p, n_classes, lstm_size, alpha
        "z": z,
        "max_len": max_len,
        "p": p,
        "lstm_size": lstm_size,
        "alpha": alpha,
        "dataset": dataset,
        "vocab": "vocab.pkl"
    })

    decay_after_num_epochs = num_epochs * 0.7

    opt.train(num_epochs, decay_after=decay_after_num_epochs, lr_decay=0.95, decay_schedule_in_iters=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', default=100, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-anneal_start', default=50000., type=float)
    parser.add_argument('-anneal_end', default=60000., type=float)
    parser.add_argument('-p', default=0.0, type=float)
    parser.add_argument('-alpha', default=0.2, type=float)
    parser.add_argument('-lstm_size', default=1000, type=int)
    parser.add_argument('-num_epochs', default=50, type=int)
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-session', type=str)
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-resume', default=False, type=bool)
    args = parser.parse_args()
    main(**vars(args))
