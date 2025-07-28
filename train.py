import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import time

# --- Load Data ---
data_file = "/path/to/your/folder_name/train_data.npy"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Training data file {data_file} not found")

data = np.load(data_file)
print(f"Loaded {len(data)} passwords of length {data.shape[1]}")

# --- Parameters ---
z_dim = 256
batch_size = 96
epochs = 25
pw_len = data.shape[1]
# Assuming the data is integer-encoded. Let's calculate vocab_size safely.
vocab_size = int(np.max(data)) + 1
steps_per_epoch = len(data) // batch_size

lambda_gp = 10.0
critic_iters = 5

# --- Keras Models for TF2 ---

# Generator Model
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(pw_len * vocab_size),
        tf.keras.layers.Reshape((pw_len, vocab_size)),
        tf.keras.layers.Softmax()
    ], name="generator")
    return model

# Critic Model
def build_critic():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(pw_len, vocab_size)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name="critic")
    return model

generator = build_generator()
critic = build_critic()

# --- Optimizers ---
g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
c_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

# --- Checkpoint Manager ---
output_dir = "/save_to_dir/passgan_output"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                 critic_optimizer=c_optimizer,
                                 generator=generator,
                                 critic=critic)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


# --- Loss Functions ---
def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# --- Gradient Penalty ---
def gradient_penalty(real_images, fake_images):
    alpha = tf.random.normal([real_images.shape[0], 1, 1], 0.0, 1.0)
    interpolated = real_images + alpha * (fake_images - real_images)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0)**2)
    return gp

# --- Training Step Function (`tf.function` for performance) ---
@tf.function
def train_step(real_images, noise):
    # --- Critic Training ---
    for _ in range(critic_iters):
        with tf.GradientTape() as c_tape:
            fake_images = generator(noise, training=True)
            real_output = critic(real_images, training=True)
            fake_output = critic(fake_images, training=True)

            c_loss = critic_loss(real_output, fake_output)
            gp = gradient_penalty(real_images, fake_images)
            total_c_loss = c_loss + lambda_gp * gp

        c_gradients = c_tape.gradient(total_c_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(c_gradients, critic.trainable_variables))

    # --- Generator Training ---
    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_output = critic(fake_images, training=True)
        g_loss = generator_loss(fake_output)

    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return total_c_loss, g_loss

# --- Decoder ---
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ ")
idx2char = {i + 1: c for i, c in enumerate(chars)}
idx2char[0] = ''

def decode_password(encoded):
    return ''.join(idx2char.get(i, '') for i in encoded)

# --- Training Loop ---
d_losses, g_losses = [], []
train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(batch_size, drop_remainder=True)

for epoch in range(epochs):
    start_time = time.time()

    for real_batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}"):
        real_batch_one_hot = tf.one_hot(real_batch, depth=vocab_size)
        noise = tf.random.normal([batch_size, z_dim])
        d_loss_val, g_loss_val = train_step(real_batch_one_hot, noise)

    epoch_duration = time.time() - start_time
    d_losses.append(d_loss_val.numpy())
    g_losses.append(g_loss_val.numpy())

    print(f"Epoch {epoch+1} | D_loss: {d_loss_val:.4f} | G_loss: {g_loss_val:.4f} | Time: {epoch_duration:.2f}s")

    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        ckpt_manager.save()
        print(f"Checkpoint saved for epoch {epoch + 1}")

# --- Generate and Save Passwords ---
noise = tf.random.normal([100, z_dim])
samples = generator(noise, training=False)
preds = np.argmax(samples.numpy(), axis=2)

with open(os.path.join(output_dir, "generated_passwords.txt"), "w") as f:
    for p in preds:
        f.write(decode_password(p) + "\n")
print("Generated passwords saved.")


# --- Plot and Save Losses ---
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label='Critic Loss')
plt.plot(g_losses, label='Generator Loss')
plt.legend()
plt.title("WGAN-GP Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, "final_loss_curve.png"))
plt.show()
