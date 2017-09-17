class Config(object):
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 400
  max_epoch = 10
  max_max_epoch = 30
  #keep_prob = 1
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 16000

  def __init__(self, is_training):
    self.is_training = is_training
    if not is_training:
      self.num_steps = 1
      self.batch_size = 1
