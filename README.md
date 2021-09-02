# Perceptron
Machin Learning Algorithm : Perceptron

You will implement several variants of the Perceptron algorithm. Note that each variant
has different hyper-parameters, as described below. Use 5-fold cross-validation to identify
the best hyper-parameters as you did in the previous homework. To help with this, we have
split the training set into five parts fold1–fold5 in the folder CVFolds (inside both the
csv-format/ and libSVM-format/ directories).
(If you use a random number generator to initialize weights or shuffle data during training,
please set the random seed in the beginning to a fixed number so that different runs produce
the same results. Each language has its own way to set the random seed. For example, in
python, you can set the seed to be, say the number 42, using random.seed(42).)

this code include: 

1. Simple Perceptron
2. Decaying the learning rate
3. Average Perceptron
4. Margin Perceptron
