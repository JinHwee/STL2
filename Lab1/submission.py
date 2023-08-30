from math import ceil

# global variables for the script, after obtaining partial decrypted cipher algorithm
RULE = [(101 >> i) & 1 for i in range(8)]
N_BYTES = 32
N = 8 * N_BYTES

extensions = {
    "0" : ["001", "011", "100", "111"],
    "1" : ["000", "010", "101", "110"]
}

def next(x):
    x =  (x & 1) << N+1 | x << 1 | x >> N-1
    y = 0
    for i in range(N):
        y |= RULE[(x >> i) & 7] << i
    return y

# Qns: Retrieving partial keystream from plaintext and ciphertext
def retrieve_partial_key():

    with open('csdp.txt', 'rb') as file:
        plaintext = file.readline()
        file.close()

    with open('csdp.txt.enc', 'rb') as file:
        ciphertext = file.readline()
        file.close()

    partial_key = []
    length = min(len(plaintext), len(ciphertext))
    for i in range(length):
        partial_key.append(plaintext[i] ^ ciphertext[i])
    
    partial_key = bytes(partial_key)
    return partial_key

# decrypting keystream to retrieve plaintext seed
def prev(stream):
    stream = "{0:b}".format(stream)
    stream = stream.zfill(N)

    possible_strings = extensions[stream[-1]]
    for i in reversed(stream[:-1]):
        new_combinations = []
        extension = extensions[i]
        for ext in extension:
            for string in possible_strings:
                if ext[1:] == string[:2]:
                    new_combinations += [ext[0] + string]
        possible_strings = new_combinations

    feasible_ones = []
    for string in possible_strings:
        if string[0] == string[-2] and string[1] == string[-1]:
            feasible_ones += [string]

    seed = int(feasible_ones[0], 2)
    seed = ((seed >> 1) & ((2**N) - 1)) | ((seed & 1) << (N - 1))

    return seed

# obtaining initial seed to generate the keystream
def retrieve_initial_seed(keystream):
    keystream = keystream[0:N_BYTES]
    value = int.from_bytes(keystream, 'little')
    for i in range(N//2):
        value = prev(value)

    seed = value.to_bytes(N_BYTES, 'little')
    return seed

# generating full keystream to decrypt each file
def generate_full_keystream(seed, total_length):
    value = int.from_bytes(seed, 'little')
    for _ in range(N//2):
        value = next(value)

    keystream = int.to_bytes(value, N_BYTES, 'little')
    for _ in range(ceil(total_length/N_BYTES) - 1):
        value = next(value)
        keystream += value.to_bytes(N_BYTES, 'little')
    
    return keystream

# decrypting the files
def decrypt_files(seed):
    with open('super_cipher.py.enc', 'rb') as file:
        cipher = file.readlines()
        cipher = b''.join(cipher)
        file.close()
    
    total_length = len(cipher)
    keystream = generate_full_keystream(seed, total_length)
    super_cipher_array = []
    for i in range(len(cipher)):
        super_cipher_array.append(keystream[i] ^ cipher[i])
    super_cipher_array = bytes(super_cipher_array)
    
    with open('super_cipher_decrypted.py', 'wb') as file:
        file.write(super_cipher_array)
        file.close()

    with open('hint.gif.enc', 'rb') as file:
        cipher = file.readlines()
        cipher = b''.join(cipher)
        file.close()

    total_length = len(cipher)
    keystream = generate_full_keystream(seed, total_length)
    hint_gif = []
    for i in range(len(cipher)):
        hint_gif.append(keystream[i] ^ cipher[i])
    hint_gif = bytes(hint_gif)

    with open('hint.gif', 'wb') as file:
        file.write(hint_gif)
        file.close()

if __name__ == "__main__":
    partial_key = retrieve_partial_key()
    seed = retrieve_initial_seed(partial_key)
    decrypt_files(seed)
