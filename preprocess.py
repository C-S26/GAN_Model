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
    passwords = []
    count = 0
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()  # Strip whitespace
            if 1 <= len(line) <= MAX_LEN:
                passwords.append(encode_password(line))
                count += 1
            if count >= limit:  # Stop if we reach the limit
                break
    return np.array(passwords)

# Save encoded data
if __name__ == "__main__":
    data = load_data("output.txt")  # Change path if needed
    np.save("train_data.npy", data)
