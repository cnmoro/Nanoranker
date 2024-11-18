from sklearn.feature_extraction.text import HashingVectorizer
from onnxruntime_extensions import get_library_path
from multiprocessing import cpu_count
import onnxruntime as ort
import torch, importlib
import torch.nn as nn
from tqdm import tqdm
import numpy as np

reranker_nn = nn.Sequential(
    nn.Linear(1152, 1280),
    nn.ReLU(),
    nn.Linear(1280, 1536),
    nn.ReLU(),
    nn.Linear(1536, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

reranker_nn.load_state_dict(torch.load(str(importlib.resources.files('nanoranker').joinpath('resources/reranker_nn_v2_state_dict.pth')), weights_only=True))
reranker_nn.eval()

hv = HashingVectorizer(ngram_range=(1, 6), analyzer="char", n_features=64)

onnx_model_path = importlib.resources.files('nanoranker').joinpath('resources/universal-sentence-encoder-multilingual.onnx')

cpu_core_count = cpu_count() // 4
_options = ort.SessionOptions()
_options.inter_op_num_threads, _options.intra_op_num_threads = cpu_core_count, cpu_core_count
_options.register_custom_ops_library(get_library_path())
_providers = ["CPUExecutionProvider"]

embedding_model = ort.InferenceSession(
    path_or_bytes = onnx_model_path,
    sess_options=_options,
    providers=_providers
)
    
def textual_embed(text):
    return np.sum(hv.fit_transform([text]).toarray(), axis=0)


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
    )

    X = np.array(combined_embeddings).astype(np.float32)

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(X)), desc="Predicting"):
            inputs = torch.from_numpy(X[i:i+1])
            outputs = reranker_nn(inputs)
            predictions.append(outputs.squeeze().detach().numpy())

    # Merge the predictions with the documents
    results = []
    for sentence, prediction in zip(documents, predictions):
        results.append((sentence, float(prediction)))

    # Sort by the prediction value, descending
    results = sorted(results, key=lambda x: x[1], reverse=True)

    if normalize_scores:
        total_score = sum([result[1] for result in results])
        results = [(result[0], result[1] / total_score) for result in results]

    return results[:top_n]
