import tensorflow as tf


# 普通的RNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2)))

x = tf.random.normal((1, 3, 2))

layer = tf.keras.layers.SimpleRNN(4, input_shape=(3, 2))
output = layer(x)

print(output.shape)
print(output)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(3, 2))
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2)))
print(model.summary())

embedding_matrix = tf.constant(
        [[0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]],dtype=tf.float32)

tf.keras.layers.Embedding(4,
                          4,
                          embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                          trainable=True)

# embedding
embedding = tf.constant(
        [[0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]],dtype=tf.float32)

feature_batch = tf.constant([2,3,1,0])

get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)
print(get_embedding1)


# 多输出的RNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2),
                    return_sequences=True))

x = tf.random.normal((1, 3, 2))

layer = tf.keras.layers.SimpleRNN(4, input_shape=(3, 2), return_sequences=True)
output = layer(x)

print(output.shape)
print(output)

# 每个时间步增加层
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2),
                    return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax')))

# 多层叠加
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2), return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(4, input_shape=(3, 2), return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(4))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10)))
model.add(tf.keras.layers.Dense(5))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print(model.summary())


# LSTM
inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)

lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
out, h_state, c_state = lstm(inputs)
print(out.shape)
print(h_state.shape)
print(c_state.shape)


# GRU
inputs = tf.random.normal([32, 10, 8])
gru = tf.keras.layers.GRU(4)
output = gru(inputs)
print(output.shape)

gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
out, final_state = gru(inputs)
print(out.shape)
print(final_state.shape)


# Seq2seq
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, state, attention_weights
