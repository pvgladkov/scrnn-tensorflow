# Structurally Constrained Recurrent Neural Network

TensorFlow implementation of SCRNN cell from the paper titled: Learning Longer Memory in Recurrent Neural Networks: http://arxiv.org/abs/1412.7753.

Original torch implementation [https://github.com/facebookarchive/SCRNNs](https://github.com/facebookarchive/SCRNNs).


## Results

|cell|num epoch|size|train perplexity|test perplexity|trainable parameters|
|---|---|---|---|---|---|
|LSTM|100|128|2.473|4.052|97457|
|LSTM|1000|128|1.004|1.004|97457|
|SCRN|100|128|5.774|7.590|55121|
|SCRN|1000|128|1.882|2.816|55121|
|SCRN|2500|128|1.004|1.006|55121|
|BasicRNN|100|128|17.391|20.973|29105|
|BasicRNN|1000|128|1.218|2.370|29105|
|GRU|100|128|1.314|2.942|74673|
|GRU|1000|128|1.001|1.001|74673|

### SCRN

![scrn perplexity](https://raw.githubusercontent.com/webgeist/scrnn-tensorflow/master/results/scrn-128-32-005-2500.png)

### LSTM

![lstm preplexity](https://raw.githubusercontent.com/webgeist/scrnn-tensorflow/master/results/lstm-128-32-005-2000.png)

### GRU

![gru preplexity](https://raw.githubusercontent.com/webgeist/scrnn-tensorflow/master/results/gru-128-32-005-2000.png)

### Basic RNN

![rnn preplexity](https://raw.githubusercontent.com/webgeist/scrnn-tensorflow/master/results/rnn-128-32-005-1500.png)

### Get data
```
$ bash ./data/makedata-ptb.sh
```

## Usage

```
$ python train.py --cell scrnn --lr 0.1 --num_epoch 250
```

### Args

`cell` - type of rnn cell. `lstm` or `scrnn`.

`lr` - learning rate.

`seq_length` - sequence length.

`rnn_size` - size of the hidden layer.

`num_epoch` - number of epochs.

`batch_size` - batch size. 

