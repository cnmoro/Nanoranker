from onnxruntime_extensions import get_library_path
from multiprocessing import cpu_count
import onnxruntime as ort
import importlib.resources
import numpy as np
import re

# ---------------------------------------------------------------------------
# Textual embedding: a pure-numpy re-implementation of scikit-learn's
# HashingVectorizer(ngram_range=(1, 6), analyzer="char", n_features=64).
# Validated to be bit-for-bit identical to sklearn's output, which lets us
# drop the scikit-learn dependency entirely.
# ---------------------------------------------------------------------------

_UINT32_MASK = 0xFFFFFFFF


def _murmurhash3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 x86_32, returning a signed 32-bit int (matches sklearn)."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    length = len(data)
    h1 = seed
    rounded_end = length & ~0x03
    for i in range(0, rounded_end, 4):
        k1 = (data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)) & _UINT32_MASK
        k1 = (k1 * c1) & _UINT32_MASK
        k1 = ((k1 << 15) | (k1 >> 17)) & _UINT32_MASK
        k1 = (k1 * c2) & _UINT32_MASK
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & _UINT32_MASK
        h1 = (h1 * 5 + 0xE6546B64) & _UINT32_MASK

    k1 = 0
    tail = length & 0x03
    if tail == 3:
        k1 ^= data[rounded_end + 2] << 16
    if tail >= 2:
        k1 ^= data[rounded_end + 1] << 8
    if tail >= 1:
        k1 ^= data[rounded_end]
        k1 = (k1 * c1) & _UINT32_MASK
        k1 = ((k1 << 15) | (k1 >> 17)) & _UINT32_MASK
        k1 = (k1 * c2) & _UINT32_MASK
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _UINT32_MASK
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _UINT32_MASK
    h1 ^= h1 >> 16
    return h1 - 0x100000000 if h1 & 0x80000000 else h1


_WHITESPACE = re.compile(r"\s\s+")
_N_FEATURES = 64
_MIN_N, _MAX_N = 1, 6


def _char_ngrams(text):
    """Character n-grams (1..6), mirroring sklearn's char analyzer."""
    text = _WHITESPACE.sub(" ", text)
    tlen = len(text)
    for n in range(_MIN_N, _MAX_N + 1):
        if n > tlen:
            break
        for i in range(tlen - n + 1):
            yield text[i:i + n]


def textual_embed(text):
    vec = np.zeros(_N_FEATURES, dtype=np.float64)
    for token in _char_ngrams(text.lower()):
        h = _murmurhash3_32(token.encode("utf-8"), 0)
        vec[abs(h) % _N_FEATURES] += 1.0 if h >= 0 else -1.0
    norm = np.sqrt((vec * vec).sum())
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# ONNX runtime sessions
# ---------------------------------------------------------------------------

_cpu_core_count = max(1, cpu_count() // 4)
_options = ort.SessionOptions()
_options.inter_op_num_threads = _cpu_core_count
_options.intra_op_num_threads = _cpu_core_count
_options.register_custom_ops_library(get_library_path())
_providers = ["CPUExecutionProvider"]

_resources = importlib.resources.files("nanoranker").joinpath("resources")

embedding_model = ort.InferenceSession(
    path_or_bytes=str(_resources.joinpath("universal-sentence-encoder-multilingual.onnx")),
    sess_options=_options,
    providers=_providers,
)

reranker_model = ort.InferenceSession(
    path_or_bytes=str(_resources.joinpath("reranker_nn_v2.onnx")),
    sess_options=_options,
    providers=_providers,
)


def semantic_embed(text):
    return embedding_model.run(output_names=["outputs"], input_feed={"inputs": [text]})[0][0]


def embed(text):
    return np.concatenate([semantic_embed(text), textual_embed(text)])


def rank(query, documents, top_n=10, normalize_scores=True):
    query_embedding = embed(query)
    sentence_embeddings = np.array([embed(sentence) for sentence in documents])

    combined_embeddings = np.concatenate(
        [
            np.tile(query_embedding, (len(sentence_embeddings), 1)),
            sentence_embeddings,
        ],
        axis=1,
    ).astype(np.float32)

    predictions = reranker_model.run(["output"], {"input": combined_embeddings})[0].reshape(-1)

    results = sorted(
        zip(documents, (float(p) for p in predictions)),
        key=lambda x: x[1],
        reverse=True,
    )

    if normalize_scores:
        total_score = sum(score for _, score in results)
        if total_score:
            results = [(doc, score / total_score) for doc, score in results]

    return results[:top_n]
