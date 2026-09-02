"""Tokenizer byte-level: cada token es exactamente un byte (0..255)."""
VOCAB_SIZE = 256

def encode(text: str):
    return list(text.encode('utf-8'))

def decode(tokens):
    return bytes(int(x) % 256 for x in tokens).decode('utf-8', errors='replace')
