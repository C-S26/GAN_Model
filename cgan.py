import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Load Data ---
data_file = "/content/train_data (2).npy"
try:
    data = np.load(data_file)
    print(f"Successfully loaded training data from {data_file}")
except FileNotFoundError:
    print(f"Error: Training data file '{data_file}' not found!")
    print("Please ensure the training data file exists at the specified path.")
    raise FileNotFoundError(f"Training data file not found: {data_file}")
except Exception as e:
    print(f"Error loading training data: {e}")
    raise

print(f"Loaded {len(data)} passwords of length {data.shape[1]}")

# --- Create synthetic condition labels (e.g., password length bucket) ---
pw_len = data.shape[1]
vocab_size = int(np.max(data)) + 1

# Simulate 3 conditions: short, medium, long
length_classes = np.array([0 if l <= 8 else 1 if l <= 12 else 2 for l in np.sum(data > 0, axis=1)])
num_classes = 3  # short, medium, long

# One-hot encode labels
conditions = tf.keras.utils.to_categorical(length_classes, num_classes=num_classes)

# --- FIXED Hyperparameters for Better Training ---
z_dim = 64  # Reduced from 128
batch_size = min(32, len(data) // 20)  # Smaller batches for stability
epochs = 50  # More epochs with better learning
lambda_gp = 10.0  # Standard gradient penalty
critic_iters = 2  # Reduced for better balance

print(f"Training parameters: batch_size={batch_size}, epochs={epochs}")

# --- Define IMPROVED Generator with Better Architecture ---
def build_generator():
    z_input = tf.keras.layers.Input(shape=(z_dim,))
    label_input = tf.keras.layers.Input(shape=(num_classes,))

    # Better label integration - expand to match noise dimension
    label_expanded = tf.keras.layers.Dense(z_dim//4, activation='relu')(label_input)
    label_expanded = tf.keras.layers.BatchNormalization()(label_expanded)
    
    x = tf.keras.layers.Concatenate()([z_input, label_expanded])

    # Simpler, more stable architecture
    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    x = tf.keras.layers.Dense(512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Final layers
    x = tf.keras.layers.Dense(pw_len * vocab_size)(x)
    x = tf.keras.layers.Reshape((pw_len, vocab_size))(x)
    output = tf.keras.layers.Softmax(axis=-1)(x)
    
    return tf.keras.Model([z_input, label_input], output, name="generator")

# --- Define IMPROVED Discriminator ---
def build_discriminator():
    pw_input = tf.keras.layers.Input(shape=(pw_len, vocab_size))
    label_input = tf.keras.layers.Input(shape=(num_classes,))

    # Flatten password input
    pw_flat = tf.keras.layers.Flatten()(pw_input)

    # Better label processing
    label_expanded = tf.keras.layers.Dense(64, activation='relu')(label_input)
    
    # Concatenate inputs
    x = tf.keras.layers.Concatenate()([pw_flat, label_expanded])

    # Simpler discriminator architecture - prevents overpowering generator
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(1)(x)  # No final activation for WGAN
    
    return tf.keras.Model([pw_input, label_input], x, name="discriminator")

generator = build_generator()
discriminator = build_discriminator()

print("Models built successfully")

# --- BALANCED Learning Rates ---
steps_per_epoch = max(1, len(data) // batch_size)

# More conservative learning rates
g_lr = 1e-4  # Generator learning rate
d_lr = 4e-5  # Discriminator learning rate (lower to prevent overpowering)

g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.5, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.5, beta_2=0.9)

# --- Utility ---
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ ")
idx2char = {i + 1: c for i, c in enumerate(chars)}
idx2char[0] = ''

def decode_password(encoded):
    if isinstance(encoded, tf.Tensor):
        encoded = encoded.numpy()
    return ''.join(idx2char.get(int(i), '') for i in encoded)

# --- IMPROVED Loss Functions ---
@tf.function
def discriminator_loss(real_output, fake_output):
    # Wasserstein loss
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

@tf.function
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# --- FIXED Gradient Penalty Implementation ---
@tf.function
def gradient_penalty(real_samples, fake_samples, labels):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0., 1.)
    
    # Linear interpolation between real and fake samples
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = discriminator([interpolated, labels], training=True)
    
    # Calculate gradients with respect to interpolated samples
    gradients = tape.gradient(predictions, interpolated)
    
    # Calculate gradient penalty
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-12)
    gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))
    
    return gradient_penalty

# --- IMPROVED Training Step ---
@tf.function
def train_step(real_data, labels):
    batch_size = tf.shape(real_data)[0]
    
    # --- Train Discriminator ---
    for _ in range(critic_iters):
        noise = tf.random.normal([batch_size, z_dim])
        
        with tf.GradientTape() as disc_tape:
            fake_data = generator([noise, labels], training=True)
            
            real_output = discriminator([real_data, labels], training=True)
            fake_output = discriminator([fake_data, labels], training=True)
            
            disc_loss = discriminator_loss(real_output, fake_output)
            gp = gradient_penalty(real_data, fake_data, labels)
            total_disc_loss = disc_loss + lambda_gp * gp
        
        disc_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        # Gradient clipping for stability
        disc_gradients = [tf.clip_by_norm(grad, 1.0) for grad in disc_gradients]
        d_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # --- Train Generator ---
    noise = tf.random.normal([batch_size, z_dim])
    
    with tf.GradientTape() as gen_tape:
        fake_data = generator([noise, labels], training=True)
        fake_output = discriminator([fake_data, labels], training=True)
        gen_loss = generator_loss(fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_gradients = [tf.clip_by_norm(grad, 1.0) for grad in gen_gradients]
    g_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return total_disc_loss, gen_loss

# --- Dataset Preparation ---
try:
    onehot_pw = tf.one_hot(data, depth=vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((onehot_pw, conditions))
    dataset = dataset.shuffle(len(data)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    print("Dataset prepared successfully")
except Exception as e:
    print(f"Error preparing dataset: {e}")
    raise

# --- Training Loop with Better Monitoring ---
d_losses, g_losses = [], []
best_d_loss = float('inf')
patience = 10
patience_counter = 0

print("Starting training with corrected parameters...")
print(f"Target: Stable convergence with balanced losses")

for epoch in range(epochs):
    start_time = time.time()
    d_loss_epoch, g_loss_epoch = [], []
    
    try:
        for real_pw, real_cond in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            d_loss_val, g_loss_val = train_step(real_pw, real_cond)
            
            # Check for invalid values
            if tf.math.is_finite(d_loss_val) and tf.math.is_finite(g_loss_val):
                d_loss_epoch.append(float(d_loss_val.numpy()))
                g_loss_epoch.append(float(g_loss_val.numpy()))
            else:
                print(f"Warning: Invalid loss values at epoch {epoch+1}")
                continue

        if not d_loss_epoch or not g_loss_epoch:
            print(f"No valid steps in epoch {epoch+1}")
            continue

        avg_d_loss = np.mean(d_loss_epoch)
        avg_g_loss = np.mean(g_loss_epoch)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        epoch_time = time.time() - start_time

        # Enhanced progress reporting
        print(f"Epoch {epoch+1:2d}: D_loss = {avg_d_loss:7.4f}, G_loss = {avg_g_loss:7.4f}, Time: {epoch_time:.1f}s")

        # Training stability assessment
        if len(d_losses) > 3:
            recent_d = np.mean(d_losses[-3:])
            recent_g = np.mean(g_losses[-3:])
            
            # Check for convergence
            if abs(avg_d_loss) < 0.5 and abs(avg_g_loss) < 2.0:
                print("   ✅ Training appears stable")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Balance check
            if avg_d_loss != 0:
                loss_ratio = abs(avg_g_loss / avg_d_loss)
                if 1.0 <= loss_ratio <= 5.0:
                    print(f"   ✅ Good balance (ratio: {loss_ratio:.2f})")
                else:
                    print(f"   ⚠  Imbalanced (ratio: {loss_ratio:.2f})")

        # Early stopping if training becomes unstable
        if patience_counter >= patience:
            print("   🛑 Early stopping - training appears unstable")
            break

        # --- Sample Generation ---
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"\n--- Generated samples at epoch {epoch + 1} ---")
            try:
                z_sample = tf.random.normal([6, z_dim])
                cond_sample = tf.keras.utils.to_categorical([0, 1, 2, 0, 1, 2], num_classes)
                preds = generator.predict([z_sample, cond_sample], verbose=0)
                decoded = [decode_password(np.argmax(p, axis=1)) for p in preds]
                
                condition_names = ["Short", "Medium", "Long"]
                for i, pw in enumerate(decoded):
                    print(f"  {condition_names[i % 3]:6s}: '{pw}'")
            except Exception as e:
                print(f"  Error generating samples: {e}")
            print()

    except KeyboardInterrupt:
        print("Training interrupted by user")
        break
    except Exception as e:
        print(f"Error in epoch {epoch+1}: {e}")
        continue

# --- Final Password Generation ---
if d_losses and g_losses:
    print("\n🔄 Generating final password collection...")

    output_dir = "passgan_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        num_generate = 1000
        batch_size_gen = 50
        all_passwords = []

        for i in range(0, num_generate, batch_size_gen):
            current_batch_size = min(batch_size_gen, num_generate - i)

            z = tf.random.normal([current_batch_size, z_dim])
            
            # Generate mix of all conditions
            labels = np.random.choice([0, 1, 2], size=current_batch_size)
            cond_batch = tf.keras.utils.to_categorical(labels, num_classes=3)

            generated_batch = generator.predict([z, cond_batch], verbose=0)
            decoded_batch = [decode_password(np.argmax(p, axis=1)) for p in generated_batch]

            all_passwords.extend(decoded_batch)

        # Filter and save
        valid_passwords = [pw.strip() for pw in all_passwords if len(pw.strip()) >= 4]
        
        with open(os.path.join(output_dir, "generated_passwords.txt"), "w", encoding='utf-8') as f:
            for pw in valid_passwords:
                f.write(pw + "\n")

        print(f"✅ Saved {len(valid_passwords)} valid passwords")

        # Statistics
        lengths = [len(pw) for pw in valid_passwords]
        if lengths:
            print(f"Length stats: Min={min(lengths)}, Max={max(lengths)}, Avg={np.mean(lengths):.1f}")

    except Exception as e:
        print(f"Error generating final passwords: {e}")

    # --- Visualization ---
    try:
        plt.figure(figsize=(12, 8))
        
        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(d_losses, label="Discriminator Loss", linewidth=2, color='red')
        plt.plot(g_losses, label="Generator Loss", linewidth=2, color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CGAN Training Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Loss ratio
        plt.subplot(2, 2, 2)
        if len(d_losses) > 0 and all(abs(d) > 1e-6 for d in d_losses):
            loss_ratios = [abs(g/d) for g, d in zip(g_losses, d_losses)]
            plt.plot(loss_ratios, label="|G_loss/D_loss|", linewidth=2, color='green')
            plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Perfect Balance')
            plt.xlabel("Epoch")
            plt.ylabel("Loss Ratio")
            plt.title("Training Balance")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Smoothed losses
        plt.subplot(2, 2, 3)
        if len(d_losses) > 5:
            window = 5
            d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
            g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
            plt.plot(d_smooth, label='D Loss (smoothed)', linewidth=2, color='darkred')
            plt.plot(g_smooth, label='G Loss (smoothed)', linewidth=2, color='darkblue')
            plt.xlabel("Epoch")
            plt.ylabel("Smoothed Loss")
            plt.title("Training Stability")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Training progress
        plt.subplot(2, 2, 4)
        epochs_completed = list(range(1, len(d_losses) + 1))
        plt.plot(epochs_completed, [-d for d in d_losses], label='Wasserstein Distance', linewidth=2, color='purple')
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.title("Wasserstein Distance Estimate")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "corrected_training_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training analysis saved to: {plot_path}")

    except Exception as e:
        print(f"Error creating plots: {e}")

    # --- Final Assessment ---
    print(f"\n=== FINAL TRAINING ASSESSMENT ===")
    final_d = d_losses[-1]
    final_g = g_losses[-1]
    
    print(f"Final Discriminator Loss: {final_d:.4f}")
    print(f"Final Generator Loss: {final_g:.4f}")
    print(f"Total Epochs Completed: {len(d_losses)}")
    
    if abs(final_d) < 0.5 and abs(final_g) < 2.0:
        print("🏆 Training converged successfully!")
    else:
        print("⚠  Training may need more tuning")
    
    print(f"Output Directory: {output_dir}")

else:
    print("No training data recorded - check for errors above")
