# Neural Networks with Back-Propagation Algorithm

## Features

- support any number of layers and neurons in each layer
- use numpy inside to accelerate

Note: all hidden and output neurons are with tanh-type transformation

## Usage

- data

accept plain list or numpy array.

- define NNet

```
nn = NNet([n_0, n_1, ..., n_L])
```

- training

```
nn.train(X, y)
```

- training MLS error

```
error = nn.score(X, y)
```

- predict

```
y_pred = nn.predict(X)
```

