import re
import collections
import tensorflow as tf

def split_text(content):
  content = re.sub('\s+', ' ', content)
  content = re.split(r'(?:(\w+)|(<[a-zA-Z]+>)|(\.!\?-))', content, flags=re.UNICODE)
  return [word for word in content if word != None and word != ' ' and word != '']

def read_zip_content(zf):
  total_content = []
  for filename in zf.namelist():
    if filename.endswith('.html') and filename != 'index.html':
      info = zf.getinfo(filename)
      content = zf.read(filename).decode('utf-8')
      total_content += split_text(content)
  return total_content

def build_vocabulary(data):
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id, words

def content_to_word_ids(content, word_to_id):
  return [word_to_id[word] for word in content]

def to_text(text_ids, words):
  text = ' '.join([words[word_id] for word_id in text_ids])
  return re.sub(' [,.] ', '', text)

def text_producer(raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "TextProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
