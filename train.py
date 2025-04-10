import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import trange

# Load preprocessed password data
data = np.load("train_data.npy")

# GAN params
z_dim = 100
batch_size = 64
steps = 100000
pw_len = data.shape[1]
vocab_size = np.max(data) + 1  # total tokens used

# One-hot encode inputs
def one_hot(x, depth):
    return np.eye(depth)[x]

# Generator model
def generator(z, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        h1 = tf.layers.dense(z, 256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, pw_len * vocab_size)
        out = tf.reshape(h2, [-1, pw_len, vocab_size])
        return tf.nn.softmax(out)

# Discriminator model
def discriminator(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        flat = tf.reshape(x, [-1, pw_len * vocab_size])
        h1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
        out = tf.layers.dense(h2, 1)
        return out

# Placeholders
Z = tf.placeholder(tf.float32, [None, z_dim])
X = tf.placeholder(tf.float32, [None, pw_len, vocab_size])

G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample, reuse=True)

# Losses
D_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + \
    tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

# Variables
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# Optimizers
D_train = tf.train.AdamOptimizer(1e-4).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=G_vars)

# Training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())

d_losses = []
g_losses = []

for step in trange(steps):
    idx = np.random.randint(0, len(data), batch_size)
    real_batch = one_hot(data[idx], vocab_size)
    noise = np.random.uniform(-1., 1., size=[batch_size, z_dim])

    sess.run(D_train, feed_dict={X: real_batch, Z: noise})
    sess.run(G_train, feed_dict={Z: noise})

    if step % 5000 == 0:
      d_loss_val, g_loss_val = sess.run([D_loss, G_loss], feed_dict={X: real_batch, Z: noise})
      d_losses.append(d_loss_val)
      g_losses.append(g_loss_val)
      print(f"[{step}] D_loss: {d_loss_val:.4f}, G_loss: {g_loss_val:.4f}")

# Generate passwords
def decode_password(encoded, idx2char):
    return ''.join(idx2char.get(i, '') for i in encoded)

# Build reverse index for decoding
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ ")
idx2char = {i+1: c for i, c in enumerate(chars)}
idx2char[0] = ''  # padding

print("\nGenerated Passwords:")
samples = sess.run(G_sample, feed_dict={Z: np.random.uniform(-1., 1., size=[10, z_dim])})
preds = np.argmax(samples, axis=2)
for p in preds:
    print(decode_password(p, idx2char))
with open("generated_passwords.txt", "w") as f:
    for p in preds:
        pw = decode_password(p, idx2char)
        print(pw)
        f.write(pw + "\n")
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.legend()
plt.title("PassGAN Loss Curve")
plt.show()
