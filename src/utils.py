import json
import regex as re
import numpy as np
from config import vocab_size, seq_len, d_model

# Needed for alternative data cleaning approach
# import unicodedata

def relu(x):
    return x

# TODO: Consider normalizing symbols ” and ’ to " and ' in corpus

# Clean raw data corpus
def clean(input_path = "../data/raw_corpus.txt", output_path = "../data/clean_corpus.txt"):
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    # Alternative data cleaning method
    # data = unicodedata.normalize("NFKC", data)
    # data = ''.join(c for c in data if not unicodedata.category(c).startswith('C'))

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(data)

# BPE Tokenizer
def tokenize(dataset, vocabulary_output_path='../data/vocabulary0.json', corpus_output_path='../data/tokenized_corpus_ids0.npy'):
    # Tokenize

    # Process dataset
    vocab = sorted(list(set(dataset)))

    processing_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    processed_tokens = re.findall(processing_pattern, dataset)

    tokens = [list(pt) for pt in processed_tokens]

    # TODO: Verify further that processing is working as intended
    # print(tokens[:1000])

    new_tokens = []

    while len(vocab) < vocab_size:
        # Rank pairs
        pairs = {}

        for token in tokens:
            for pair in zip(token, token[1:]):  # Possible edge case at last character?
                pairs[pair] = pairs.get(pair, 0) + 1
        
        ranked_pairs = sorted(pairs.items(), key=lambda item: item[1], reverse=True)

        # Merge
        freq_thresh = 10
    
        new_tokens = list(tokens)
        for ranked_pair in ranked_pairs:
            if ranked_pair[1] > freq_thresh and len(vocab) < vocab_size:
                pair = list(ranked_pair[0])
                merged_token = ''.join(pair)
                for idx, token in enumerate(tokens):
                    occs = [i for i in range(len(token) - len(pair) + 1) if token[i:i+len(pair)] == pair]
                    for occ in occs:
                        new_tokens[idx][occ:occ+len(pair)] = [merged_token]
                
                vocab.append(merged_token)
            else:
                break
    
    # Save vocabulary and tokenized corpus ids
    token_to_id = {token: idx for idx, token in enumerate(vocab)}

    with open(vocabulary_output_path, 'w') as f:
        json.dump(token_to_id, f)

    tokenized_corpus_flattened = [token for sublist in new_tokens for token in sublist]
    tokenized_corpus_ids = [token_to_id[token] for token in tokenized_corpus_flattened]
    tokenized_corpus_ids = np.array(tokenized_corpus_ids, dtype=np.int16)

    np.save(corpus_output_path, tokenized_corpus_ids)

    return new_tokens, vocab

# Sinusoidal positional encoder
def build_pe():
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)

    division_term = np.exp(-np.log(10000) * np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(position * division_term)
    pe[:, 1::2] = np.cos(position * division_term)

    return pe

positional_encoding_matrix = build_pe()

def encode_position(embeddings):
    # Broadcast and add to input embeddings
    return embeddings + positional_encoding_matrix[:seq_len]


# TODO: Learn more about how sinusoidal positional encoding works (https://medium.com/@pranay.janupalli/understanding-sinusoidal-positional-encoding-in-transformers-26c4c161b7cc)
# TODO: Consider refactoring the functions for positional encoding and tokenization to make them refer to the methods they're using (sinusoidal and BPE)