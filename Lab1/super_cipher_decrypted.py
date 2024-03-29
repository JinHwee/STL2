#!/usr/bin/env python3

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("key")
args = parser.parse_args()

RULE = [101 >> i & 1 for i in range(8)]
N_BYTES = 32
N = 8 * N_BYTES

def next(x):
  x = (x & 1) << N+1 | x << 1 | x >> N-1
  y = 0
  for i in range(N):
    y |= RULE[(x >> i) & 7] << i
  return y

# Bootstrap the PNRG
keystream = int.from_bytes(args.key.encode(),'little')
for i in range(N//2):
  keystream = next(keystream)

# Encrypt / decrypt stdin to stdout
plaintext = sys.stdin.buffer.read(N_BYTES)
while plaintext:
  sys.stdout.buffer.write((
    int.from_bytes(plaintext,'little') ^ keystream
  ).to_bytes(N_BYTES,'little'))
  keystream = next(keystream)
  plaintext = sys.stdin.buffer.read(N_BYTES)
  
              