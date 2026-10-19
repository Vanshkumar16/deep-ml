import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    patterns=r"([,.:;?_!\"()']|--|\s)"
    if mode=="encode":
        piece=re.split(patterns,text)
        tokens=[piece.strip() for piece in piece if piece.strip() ]
        unk_id=vocab["<|unk|>"]
        return [vocab.get(token,unk_id) for token in tokens]
    
    if mode=="decode":
        id_to_tokens={idx:token for token,idx in vocab.items()}
        tokens=[id_to_tokens[idx] for idx in text]
        decoded=" ".join(tokens)
        decoded=re.sub(r"\s+([,.?!\"()'`:;])", r"\1",decoded)
        return decoded