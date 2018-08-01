# Structurally Constrained Recurrent Neural Network

TensorFlow implementation of SCRNN cell from the paper titled: Learning Longer Memory in Recurrent Neural Networks: http://arxiv.org/abs/1412.7753.


### Get data
```
$ bash ./data/makedata-ptb.sh
```

### Results

|cell|num epoch|size|train perplexity|test perplexity|time|
|---|---|---|---|---|---|
|LSTM|100|128|2.473|4.052|-|
|SCRNN|100|128|5.774|7.590|-|
