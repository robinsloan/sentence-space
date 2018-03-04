# Welcome to sentence space

You can find an introduction to this project, with interactive demos, [here](https://www.robinsloan.com/voyages-in-sentence-space).

This is a server designed to provide a couple of interesting artifacts. The core of it is a variational autoencoder that embeds sentences into a continuous space; as a result, you can select a point anywhere in that space and get a (more or less) coherent sentence back.

Once you've established this continuous sentence space, what can you get from it?

1. *Sentence gradients*: smooth interpolations between two input sentences.
2. *Sentence neighborhoods*: clouds of alternative sentences closely related to an input sentence.

These are very weird artifacts! If you try to write a sentence gradient by hand, you'll find it's very difficult. Is it useful? Possibly not. Is it _interesting_? Definitely!

Again, you'll find a ton more context and exploration in [this post](https://www.robinsloan.com/voyages-in-sentence-space).

## Running the server

This code isn't quite turnkey, but if you're willing to tinker, you should be able to train your own models and serve your own gradients, neighborhoods, and who-knows-what-else.

The requirements are:

* Python 2.7
* Flask
* Numpy 1.12.1
* Theano 0.9 (plus Nvidia's CUDA and cudnn)
* Pandas 0.20.1
* Matplotlib 2.0.2
* [`sentencepiece`](https://github.com/google/sentencepiece) (if you want to use the included, pretrained model)
* [`wordfilter`](https://github.com/dariusk/wordfilter)

One way to get started would be to use Anaconda:

```
conda create -n sentence-space python=2.7
source activate sentence-space
conda install flask
conda install numpy=1.12.1
conda install theano=0.9.0
conda install pandas=0.20.1
conda install matplotlib=2.0.2

pip install wordfilter
pip install sentencepiece
```

If you have those requirements installed, as well as CUDA and cudnn (which is A Whole Other Thing), it _should_ be possible to run `bash serve.sh` and get a server running. If that's not the case, open an issue and let me know. I definitely want to streamline this over time, and improve this documentation as well.

Once the server is running, the API is simple:

* `/gradient?s1=Your%20first%20sentence&s2=Your%20second%20sentence`
* `/neighborhood?s1=Your%20sentence&mag=0.2`

Both endpoints return a JSON array of results. The code is currently configured to provide seven sentences in each gradient or neighborhood, but you could make that three or 128.

## Contributors

This project is forked from [`stas-semeniuta/textvae`](https://github.com/stas-semeniuta/textvae), which is the code for the paper ["A Hybrid Convolutional Variational Autoencoder for Text Generation"](https://arxiv.org/abs/1702.02390) by Stanislau Semeniuta, Aliaksei Severyn, and Erhardt Barth. I'm indebted to Semeniuta, et. al., for their skill and generosity. If I have tinkered slightly, it is because I stood on the shoulders of smart people.

I'm indebted also to [`@richardassar`](https://github.com/richardassar), whose improvements allow this server to provide results at interactive speeds.

You can find Semeniuta et. al.'s original README in (you guessed it) `ORIGINAL-README.md`.
