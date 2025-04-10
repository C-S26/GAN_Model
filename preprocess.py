import numpy as np

MAX_LEN = 10
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\ ")
CHAR2IDX = {ch: i+1 for i, ch in enumerate(CHARS)}  # index 0 = padding

def encode_password(pw):
    encoded = [CHAR2IDX.get(c, 0) for c in pw[:MAX_LEN]]
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    return encoded

def load_data(path, limit=500000):
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    passwords = [line.strip() for line in lines if 1 <= len(line.strip()) <= MAX_LEN]
    return np.array([encode_password(p) for p in passwords[:limit]])

# Save encoded data
if __name__ == "__main__":
    data = load_data("rockyou.txt")  # change path if needed
    np.save("train_data.npy", data)
