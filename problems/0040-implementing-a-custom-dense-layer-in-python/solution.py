
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
# class Dense(Layer):
# 	def __init__(self, n_units, input_shape=None):
# 		self.layer_input = None
# 		self.input_shape = input_shape
# 		self.n_units = n_units
# 		self.trainable = True
# 		self.W = None
# 		self.w0 = None

# 	def initialize(self, optimizer):
# 		# Initialize weights W, biases w0, and optimizers
# 		pass

# 	def parameters(self):
# 		# Return total number of parameters
# 		pass

# 	def forward_pass(self, X, training=True):
# 		# Compute and return the forward pass
# 		pass

# 	def backward_pass(self, accum_grad):
# 		# Compute gradients, update weights if trainable, return grad w.r.t. input
# 		pass

# 	def output_shape(self):
# 		# Return output shape tuple
# 		pass
import math
import numpy as np


class Dense(Layer):

  def __init__(self, n_units: int, input_shape: tuple = None):
    super().__init__()
    self.n_units = n_units
    self.input_shape = input_shape
    self.trainable = True
    self.W = None
    self.b = None
    self.W_opt = None
    self.b_opt = None

  def initialize(self, optimizer) -> None:
    limit = 1.0 / math.sqrt(self.input_shape[0])
    self.W = np.random.uniform(
        -limit, limit, size=(self.input_shape[0], self.n_units)
    )
    self.b = np.zeros((self.n_units,))

    self.W_opt = optimizer
    self.b_opt = optimizer

  def parameters(self) -> int:
    return self.W.size + self.b.size

  def forward_pass(self, X, training: bool = True):
    self.layer_input = X
    return np.dot(X, self.W) + self.b

  def backward_pass(self, accum_grad):
    # 1. Compute gradient w.r.t layer input using original weights
    grad_input = np.dot(accum_grad, self.W.T)

    # 2. Compute parameter gradients dL/dW and dL/db
    grad_W = np.dot(self.layer_input.T, accum_grad)
    grad_b = np.sum(accum_grad, axis=0)

    # 3. Update parameters if trainable
    if getattr(self, 'trainable', True):
      if self.W_opt:
        self.W = self.W_opt.update(self.W, grad_W)
      if self.b_opt:
        self.b = self.b_opt.update(self.b, grad_b)

    return grad_input

  def output_shape(self):
    return (self.n_units,)