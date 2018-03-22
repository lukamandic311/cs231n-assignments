import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  
  for i in range(num_train):
    sum = 0
    score = np.dot(X[i],W)
    max = np.argmax(score)

    #For numerical stability
    score -= max
    loss += - score[y[i]] + np.log(np.sum(np.exp(score)))

    for j in range(num_classes):
      softmax_score = np.exp(score[j]) / np.sum(np.exp(score))
      if j == y[i]:
        dW[:,j] += (softmax_score - 1 ) * X[i]
      else:
        dW[:,j] += softmax_score * X[i]

  loss /= num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X,W)

  #Again, for numerical stability
  scores -= np.argmax(scores,axis = 1)[...,np.newaxis]

  exp_scores = np.exp(scores)
  correct_scores = np.choose(y,scores.T)
  softmax_scores = exp_scores / np.sum(exp_scores, axis=1)[..., np.newaxis]
  

  loss = np.sum(np.log(np.sum(exp_scores,axis=1)) - correct_scores)

  
  grad = softmax_scores
  grad[range(num_train),y] += -1

  dW = np.dot(X.T,grad)




  loss /= num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  return loss, dW

