"""Tokenizer for DNA sequences using NucEL vocabulary."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

NUCEL_REPO = "FreakingPotato/NucEL"
DNA_ALPHABET = "ACGT"

_TOK = _NT_TO_ID = _MASK_ID = _UNK_ID = _PAD_ID = None


def get_tokenizer():
    global _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID
    if _TOK is not None:
        return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID

    print(f"Loading NucEL vocab from {NUCEL_REPO} ...")
    vocab_path = hf_hub_download(NUCEL_REPO, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)

    class _Tok:
        pass
    tok = _Tok()
    tok.vocab = vocab
    tok.vocab_size = len(vocab)
    tok.mask_token_id = vocab.get("⊂", 4)
    tok.pad_token_id = vocab.get("[PAD]", 0)
    tok.unk_token_id = vocab.get("[UNK]", 1)

    def _encode(text, add_special_tokens=False):
        return [vocab.get(c, vocab.get("[UNK]", 1)) for c in text.upper()]
    tok.encode = _encode

    _TOK = tok
    _UNK_ID = tok.unk_token_id
    _MASK_ID = tok.mask_token_id
    _PAD_ID = tok.pad_token_id

    nt_to_id = {}
    for nt in DNA_ALPHABET:
        ids = tok.encode(nt)
        nt_to_id[nt] = ids[0] if ids else _UNK_ID
        nt_to_id[nt.lower()] = nt_to_id[nt]
    for nt in "NnRrYyWwSsKkMmBbDdHhVv":
        nt_to_id[nt] = _UNK_ID
    _NT_TO_ID = nt_to_id

    return _TOK, _NT_TO_ID, _MASK_ID, _UNK_ID, _PAD_ID


def tokenize_nt(seq, nt_to_id, unk_id):
    return [nt_to_id.get(c, unk_id) for c in seq]
