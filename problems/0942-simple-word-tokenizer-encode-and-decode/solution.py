import re

def encode(text, vocab):
    pattern = r"([,.:;?_!\"()']|--|\s)"
    tokens = re.split(pattern, text)

    tokens = [token.strip() for token in tokens if token.strip()]

    return [vocab[token] for token in tokens]

def decode(ids, vocab):
    id_to_token = {idx: token for token, idx in vocab.items()}

    text = " ".join(id_to_token[idx] for idx in ids)

    text = re.sub(r"\s+([,.?!\"()'])", r"\1", text)

    return text