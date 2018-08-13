# Structurally Constrained Recurrent Neural Network

TensorFlow implementation of SCRNN cell from the paper titled: Learning Longer Memory in Recurrent Neural Networks: http://arxiv.org/abs/1412.7753.


### Results

|cell|num epoch|size|train perplexity|test perplexity|time|
|---|---|---|---|---|---|
|LSTM|100|128|2.473|4.052|-|
|LSTM|1000|128|1.004|1.004|-|
|SCRN|100|128|5.774|7.590|-|
|SCRN|1000|128|1.882|2.816|-|
|SCRN|2500|128|1.004|1.006|-|
|BasicRNN|100|128|17.391|20.973|-|
|BasicRNN|1000|128|1.218|2.370|-|
|GRU|100|128|1.314|2.942|-|
|GRU|1000|128|1.001|1.001|-|


### Get data
```
$ bash ./data/makedata-ptb.sh
```

### Usage

```
$ python train.py --cell scrnn --lr 0.1 --num_epoch 250
```

#### Args

`cell` - type of rnn cell. `lstm` or `scrnn`.

`lr` - learning rate.

`seq_length` - sequence length.

`rnn_size` - size of the hidden layer.

`num_epoch` - number of epochs.

`batch_size` - batch size. 


