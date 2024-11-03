import re
import json
from collections import Counter

class Tokenize:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.byte_encoder = self.build_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def build_byte_encoder(self):
        """Creates byte to unicode mapping for byte-level BPE."""
        byte_encoder = {}
        for i in range(33, 127):
            byte_encoder[i] = chr(i)
        for i in range(161, 172):
            byte_encoder[i] = chr(i)
        for i in range(174, 256):
            byte_encoder[i] = chr(i)
        for i in range(256):
            if i not in byte_encoder:
                byte_encoder[i] = chr(256 + i)
        return byte_encoder

    def bytes_to_unicode(self, text):
        """Encodes text into byte-level."""
        return ''.join(self.byte_encoder[byte] if byte in self.byte_encoder else chr(byte) for byte in text)

    def pre_tokenize(self, text):
        """Splits text into words and handles spaces."""
        text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
        return [self.bytes_to_unicode(text.encode('utf-8'))]  # Encode text into byte-level

    def get_vocab(self):
        return self.vocab

    def train_tokenizer(self, corpus):
        """Train the tokenizer on the corpus."""
        # Pre-tokenize the corpus into byte-level tokens
        tokenized_corpus = [self.pre_tokenize(text) for text in corpus]
        
        # Count all symbol pairs in the corpus
        vocab = Counter()
        for tokens in tokenized_corpus:
            for token in tokens:
                token = ' '.join(token)
                vocab[token] += 1

        # Build initial vocabulary (single characters or bytes)
        vocab_items = sorted(vocab.items(), key=lambda x: -x[1])
        self.vocab = {token: i for i, (token, _) in enumerate(vocab_items)}

        # Learn BPE merges
        self.learn_bpe(vocab)

    def learn_bpe(self, vocab):
        """Learn the BPE merges."""
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)

    def get_stats(self, vocab):
        """Get symbol pair frequencies."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Merge the most frequent pair into a single symbol."""
        new_vocab = {}
        bigram = ' '.join(pair)
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def tokenize(self, text):
        """Tokenizes a given text using byte-level BPE."""
        tokens = []
        text = self.pre_tokenize(text)[0]
        token = ' '.join(text)
        for merge in self.merges:
            token = token.replace(' '.join(merge), ''.join(merge))
        tokens.append(self.vocab[token])
        return tokens

    def save_vocab(self, filename):
        """Save the vocabulary and merges."""
        with open(filename, 'w') as f:
            json.dump({
                "vocab": {str(k): v for k, v in self.vocab.items()},
                "merges": self.merges
            }, f)

# Example usage
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        "Byte Pair Encoding is efficient.",
        "Tokenization helps with text generation."
    ]
    
    # Instantiate tokenizer
    tokenizer = Tokenize(vocab_size=50)
    
    # Train tokenizer
    tokenizer.train_tokenizer(corpus)
    
    # Tokenize a sample text
    sample_text = "Encoding is useful."
    tokenized_output = tokenizer.tokenize(sample_text)
    
    # Print results
    print(f"Tokenized text: {tokenized_output}")
    
    # Save vocabulary
    tokenizer.save_vocab("tokenizer_vocab.json")


