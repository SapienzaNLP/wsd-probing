from transformers import AutoTokenizer, AutoModel
from nltk.corpus import wordnet as wn
from utils.wsd_utils import read_from_raganato
import numpy as np
import torch
from tqdm import tqdm
import sentencepiece
from transformers import DebertaV2Model, DebertaV2Config, DebertaV2TokenizerFast

DATA_PATH = "./semcor_data/semcor.data.xml"
KEYS_PATH = "./semcor_data/semcor.gold.key.txt"


# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name, output_hidden_states=True)


def find_positions(lst, target_number):
    positions = []
    for index, number in enumerate(lst):
        if number == target_number:
            positions.append(index)
    return positions


def extract_target_sentences(target_synset):
    count = 1
    target_sentences = []
    for _, _, wsd_sentence in read_from_raganato(DATA_PATH, KEYS_PATH):
        sentence_tokens = [
            wsd_instance.annotated_token.text for wsd_instance in wsd_sentence
        ]

        for k, wsd_instance in enumerate(wsd_sentence):
            labels = wsd_instance.labels
            if labels is not None:
                for label in labels:
                    lemma = wn.lemma_from_key(label)
                    synset = lemma.synset()
                    if str(synset) == target_synset:
                        target_sentences.append(
                            {"sentence": sentence_tokens, "target_word_position": k}
                        )
                        print(
                            f"{count}: Sentence: {sentence_tokens}, Target word position: {k}, Target word: {sentence_tokens[k]}"
                        )
                        count += 1

    return target_sentences


def extract_target_sentences_from_synsets_list(target_synsets: list) -> dict:
    res = {}
    for _, _, wsd_sentence in read_from_raganato(DATA_PATH, KEYS_PATH):
        sentence_tokens = [
            wsd_instance.annotated_token.text for wsd_instance in wsd_sentence
        ]

        for k, wsd_instance in enumerate(wsd_sentence):
            labels = wsd_instance.labels
            if labels is not None:
                for label in labels:
                    lemma = wn.lemma_from_key(label)
                    synset = lemma.synset()
                    if str(synset) in target_synsets:
                        if str(synset) in res:
                            res[str(synset)].append(
                                {
                                    "sentence": " ".join(sentence_tokens),
                                    "target_word_position": k,
                                }
                            )
                        else:
                            res[str(synset)] = [
                                {
                                    "sentence": " ".join(sentence_tokens),
                                    "target_word_position": k,
                                }
                            ]

    return res


def extract_target_sentences_from_lemmas_list(target_lemmas: list) -> dict:
    """
    Args:
        target_lemmas: list of target lemmas

    Returns:
        A dictionary of the form {"lemma": [{"sentence": sentence, "target_word_position": target_word_position}]}
    """
    res = {}
    for _, _, wsd_sentence in read_from_raganato(DATA_PATH, KEYS_PATH):
        sentence_tokens = [
            wsd_instance.annotated_token.text for wsd_instance in wsd_sentence
        ]
        for k, wsd_instance in enumerate(wsd_sentence):
            labels = wsd_instance.labels
            if labels is not None:
                for label in labels:
                    lemma = wn.lemma_from_key(label)
                    if str(lemma) in target_lemmas:
                        if str(lemma) in res:
                            res[str(lemma)].append(
                                {
                                    "sentence": sentence_tokens,
                                    "target_word_position": k,
                                }
                            )
                        else:
                            res[str(lemma)] = [
                                {
                                    "sentence": sentence_tokens,
                                    "target_word_position": k,
                                }
                            ]

    return res


def add_sentences_for_each_lemma(
    lemmas_grouped_per_lemmapos: dict, all_sentences: dict, K: int, DEBUG_EXACT_K: bool
) -> dict:
    """
    Args:
        lemmas_grouped_per_synset: dictionary of the form {"lemmapos": [lemmas]}
        all_sentences: dictionary of the form {"lemma": "sentence"}

    Returns:
        A dictionary of the form {"lemmapos": {"lemma": ["sentence", "target_word_position"]]]}}
    """
    res = {}
    for lemmapos in lemmas_grouped_per_lemmapos.keys():
        res[lemmapos] = {}
        lemmas = lemmas_grouped_per_lemmapos[lemmapos]
        for lemma in lemmas:
            if DEBUG_EXACT_K:
                res[lemmapos][lemma] = all_sentences[lemma][:K]
            else:
                res[lemmapos][lemma] = all_sentences[lemma]

    return res


def compute_corpus_embeddings(
    split: dict, TRANSFORMER_MODEL: str, LAYERS_LIST: list
) -> dict:
    """
    Function to compute the embeddings of the corpus passed as parameter

    Args:
        split: corpus of the form {"lemmapos": {"lemma": [{"sentence": sentence, "target_word_position": target_word_position}]}}

    Returns:
        A dictionary of the form {"lemmapos": {"lemma": [{"sentence": sentence, "target_word_position": target_word_position, "embedding": embedding}]}}
    """
    if TRANSFORMER_MODEL == "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_MODEL, add_prefix_space=True
        )
    elif TRANSFORMER_MODEL == "microsoft/deberta-v3-base":
        tokenizer = DebertaV2TokenizerFast.from_pretrained(
            TRANSFORMER_MODEL,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    if TRANSFORMER_MODEL == "microsoft/deberta-v3-base":
        model = DebertaV2Model.from_pretrained(
            TRANSFORMER_MODEL, output_hidden_states=True
        )
    else:
        model = AutoModel.from_pretrained(TRANSFORMER_MODEL, output_hidden_states=True)

    pbar = tqdm(split.keys())
    for lemmapos in pbar:
        for lemma in split[lemmapos].keys():
            pbar.set_description(f"Computing embedding for {lemmapos} - {lemma}")
            sentences = split[lemmapos][lemma]
            for sentence in sentences:
                sentence["embedding"] = compute_embedding(
                    sentence["sentence"],
                    sentence["target_word_position"],
                    model=model,
                    tokenizer=tokenizer,
                    LAYERS_LIST=LAYERS_LIST,
                )
    return split


def compute_embedding(
    sentence,
    target_word_position,
    model,
    tokenizer,
    LAYERS_LIST,
    is_split_into_words=True,
):
    """
    Function that compute the embedding of the target word in a sentence for each layer in the LAYERS_LIST

    Args:
        sentence: sentence to compute the embedding
        target_word_position: position of the target word in the sentence
        LAYERS_LIST: list of layers to compute the embedding
        is_split_into_words: if True, the sentence is split into words

    Returns:
        A list of dictionaries of the form {"layer": layer, "embedding": embedding}
    """
    try:
        tokenized_sentence = tokenizer(
            sentence,
            is_split_into_words=is_split_into_words,
            return_tensors="pt",
        )

        sentence_embedding = model(**tokenized_sentence)
    except:
        print(f"Error while computing embedding for sentence: {sentence}")
        return None

    word_ids = tokenized_sentence.word_ids()
    target_subwords = find_positions(word_ids, target_word_position)

    token_embeddings = []
    for layer in LAYERS_LIST:
        layer_embedding_list = []
        for subword in target_subwords:
            layer_embedding_list.append(
                torch.stack([sentence_embedding.hidden_states[layer]], dim=0)
                .sum(dim=0)[0][subword]
                .detach()
                .numpy()
            )
        token_embeddings.append(np.mean(layer_embedding_list, axis=0))

    return [
        {"layer": layer, "embedding": token_embeddings[layer]} for layer in LAYERS_LIST
    ]


def compute_training_centroids(train_word_embeddings: dict) -> dict:
    centroids = {}
    for lemma in train_word_embeddings.keys():
        centroids[lemma] = np.mean(
            [elem["embedding"] for elem in train_word_embeddings[lemma]], axis=0
        )


def compute_corpus_centroids(corpus_embeddings: dict) -> dict:
    centroids = {}
    centroids_variances = {}
    centroids_std = {}
    for lemmapos in corpus_embeddings.keys():
        for lemma in corpus_embeddings[lemmapos].keys():
            centroids[lemma] = np.mean(
                [elem["embedding"] for elem in corpus_embeddings[lemmapos][lemma]],
                axis=0,
            )
            centroids_variances[lemma] = np.var(
                [elem["embedding"] for elem in corpus_embeddings[lemmapos][lemma]],
                axis=0,
            )
            centroids_std[lemma] = np.std(
                [elem["embedding"] for elem in corpus_embeddings[lemmapos][lemma]],
                axis=0,
            )
    return centroids, centroids_variances, centroids_std


def get_nearest_centroid(
    word_embedding: list, centroids: list, DISTANCE_FUNCTION: str
) -> str:
    if DISTANCE_FUNCTION == "euclidean":
        min_distance = 1e10
    elif DISTANCE_FUNCTION == "cosine":
        min_distance = -1e10

    correct_centroid_lemma = ""
    for centroid in centroids:
        if DISTANCE_FUNCTION == "euclidean":
            distance = np.linalg.norm(word_embedding - centroid["centroid"])
        elif DISTANCE_FUNCTION == "cosine":
            distance = np.dot(word_embedding, centroid["centroid"]) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(centroid["centroid"])
            )

        if DISTANCE_FUNCTION == "euclidean":
            if distance < min_distance:
                min_distance = distance
                correct_centroid_lemma = centroid["lemma"]
        elif DISTANCE_FUNCTION == "cosine":
            if distance > min_distance:
                min_distance = distance
                correct_centroid_lemma = centroid["lemma"]

    return {"nearest_centroid": correct_centroid_lemma, "distance": min_distance}
