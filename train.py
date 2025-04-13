import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

# Check if data exists
data_file = "train_data.npy"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Training data file {data_file} not found")

# Load preprocessed password data
print(f"Loading data from {data_file}...")
data = np.load(data_file)
print(f"Loaded {len(data)} passwords of length {data.shape[1]}")

# GAN params
z_dim = 256
batch_size = 64
epochs = 10
steps_per_epoch = len(data) // batch_size
pw_len = data.shape[1]
vocab_size = int(np.max(data) + 1)  # total tokens used
print(f"Vocabulary size: {vocab_size}")

# One-hot encode inputs
def one_hot(x, depth):
    return np.eye(depth)[x]

# Generator model
def generator(z, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        h1 = tf.layers.dense(z, 512, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, 1024, activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, pw_len * vocab_size)
        out = tf.reshape(h3, [-1, pw_len, vocab_size])
        return tf.nn.softmax(out)

# Discriminator model (with dropout)
def discriminator(x, reuse=False, training=True):
    with tf.variable_scope("discriminator", reuse=reuse):
        flat = tf.reshape(x, [-1, pw_len * vocab_size])
        h1 = tf.layers.dense(flat, 512, activation=tf.nn.leaky_relu)
        h1 = tf.layers.dropout(h1, rate=0.3, training=training)
        h2 = tf.layers.dense(h1, 256, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dropout(h2, rate=0.3, training=training)
        out = tf.layers.dense(h2, 1)
        return out

# Placeholders
Z = tf.placeholder(tf.float32, [None, z_dim], name="Z")
X = tf.placeholder(tf.float32, [None, pw_len, vocab_size], name="X")
is_training = tf.placeholder(tf.bool, name="is_training")

# Model
G_sample = generator(Z)
D_real = discriminator(X, training=is_training)
D_fake = discriminator(G_sample, reuse=True, training=is_training)

# Losses
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

# Variables
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# Optimizers
D_train = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(G_loss, var_list=G_vars)

# Create output directory
output_dir = "passgan_output"
os.makedirs(output_dir, exist_ok=True)

# Build character mapping for decoding
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ ")
idx2char = {i+1: c for i, c in enumerate(chars)}
idx2char[0] = ''  # padding

def decode_password(encoded, idx2char):
    return ''.join(idx2char.get(i, '') for i in encoded)

# Training session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Create saver
saver = tf.train.Saver(max_to_keep=3)
checkpoint_path = os.path.join(output_dir, "passgan_model")

d_losses = []
g_losses = []

# Training loop
try:
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Progress bar
        t = tqdm(range(steps_per_epoch))
        epoch_d_losses = []
        epoch_g_losses = []
        
        for step in t:
            # Sample real data and noise
            idx = np.random.randint(0, len(data), batch_size)
            real_batch = one_hot(data[idx], vocab_size)
            noise = np.random.normal(0, 1, size=[batch_size, z_dim])
            
            # Train discriminator
            _, d_loss_val = sess.run(
                [D_train, D_loss], 
                feed_dict={X: real_batch, Z: noise, is_training: True}
            )
            
            # Train generator (twice to balance training)
            for _ in range(2):
                _, g_loss_val = sess.run(
                    [G_train, G_loss], 
                    feed_dict={Z: noise, is_training: True}
                )
            
            # Update progress bar
            t.set_description(f"D: {d_loss_val:.4f}, G: {g_loss_val:.4f}")
            
            epoch_d_losses.append(d_loss_val)
            epoch_g_losses.append(g_loss_val)
        
        # Average losses for the epoch
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        print(f"Epoch {epoch+1} - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
        
        # Generate sample passwords after each epoch
        samples = sess.run(G_sample, feed_dict={Z: np.random.normal(0, 1, size=[10, z_dim]), is_training: False})
        preds = np.argmax(samples, axis=2)
        
        print("\nSample passwords:")
        for p in preds[:5]:  # Show first 5 samples
            pw = decode_password(p, idx2char)
            print(pw)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            save_path = saver.save(sess, checkpoint_path, global_step=epoch+1)
            print(f"Model saved: {save_path}")
            
        # Plot and save losses after each epoch
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.legend()
        plt.title("PassGAN Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

except KeyboardInterrupt:
    print("Training interrupted!")

# Generate final passwords
print("\nGenerating final passwords...")
num_samples = 100
samples = sess.run(G_sample, feed_dict={Z: np.random.normal(0, 1, size=[num_samples, z_dim]), is_training: False})
preds = np.argmax(samples, axis=2)

# Save generated passwords
output_file = os.path.join(output_dir, "generated_passwords.txt")
with open(output_file, "w") as f:
    for p in preds:
        pw = decode_password(p, idx2char)
        f.write(pw + "\n")

print(f"Generated {num_samples} passwords saved to {output_file}")

# Plot final loss curves
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.legend()
plt.title("PassGAN Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, "final_loss_curve.png"))
plt.show()

# Close session
sess.close()
