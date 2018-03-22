import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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

  #Initialize gradient and loss to zero
  dW = np.zeros(W.shape) 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  #  SVM loss is defined as: 
  #  Li =  sum j != yi  max(0, w^T_j*xi - w^T_yi*xi + 1)

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_greater_margin = 0

    for j in xrange(num_classes):
      #Skip correct class
      if j == y[i]:
        continue
      #For every other class, calculate the sum and gradient
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        num_greater_margin += 1
        loss += margin
        dW[:, j] = dW[:, j] + X[i, :]

  #Correct class gradient
    dW[:, y[i]] = dW[:, y[i]] - X[i, :]*num_greater_margin   
  
  #Average loss.
  loss /= num_train

  #Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW = dW / num_train + 2*reg *W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  class_scores = np.dot(X,W)
  
  correct_class_scores = np.choose(y, class_scores.T)

  #Loss
  mask = np.ones(class_scores.shape, dtype=bool)
  mask[range(class_scores.shape[0]), y] = False
  wrong_class_scores = class_scores[mask].reshape(class_scores.shape[0],class_scores.shape[1]-1)
  margins = wrong_class_scores - correct_class_scores[...,np.newaxis] + 1
  margins[margins < 0] = 0
  loss = np.sum(margins) / num_train + reg * np.sum(W * W)

  #Gradient
  
  margins = class_scores - correct_class_scores[...,np.newaxis] + 1
  mask = (margins > 0 ).astype(float)
  #Correct class always has margin 1, so it is counted in mask
  num_scores_greater_zero = mask.sum(1) - 1
  mask[range(num_train),y] = -num_scores_greater_zero
  dW = np.dot(X.T,mask)

  dW = dW / num_train + 2 * reg * W



  return loss, dW
