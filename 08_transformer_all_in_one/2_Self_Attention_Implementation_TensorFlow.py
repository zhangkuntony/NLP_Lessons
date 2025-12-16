import tensorflow as tf

x = [
    [1, 0, 1, 0],               # Input 1
    [0, 2, 0, 2],               # Input 2
    [1, 1, 1, 1]                # Input 3
]

x = tf.convert_to_tensor(x, dtype=tf.float32)
print(x)

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = tf.convert_to_tensor(w_key, dtype=tf.float32)
w_query = tf.convert_to_tensor(w_query, dtype=tf.float32)
w_value = tf.convert_to_tensor(w_value, dtype=tf.float32)
print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)


keys = tf.matmul(x, w_key)
queries = tf.matmul(x, w_query)
values = tf.matmul(x, w_value)
print(keys)
print(queries)
print(values)

attn_scores = tf.matmul(queries, keys, transpose_b=True)
print(attn_scores)

attn_scores_softmax = tf.nn.softmax(attn_scores, axis=-1)
print(attn_scores_softmax)

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = tf.convert_to_tensor(attn_scores_softmax)
print(attn_scores_softmax)

weighted_values = values[:,None] * tf.transpose(attn_scores_softmax)[:,:,None]
print(weighted_values)

outputs = tf.reduce_sum(weighted_values, axis=0) 
print(outputs)