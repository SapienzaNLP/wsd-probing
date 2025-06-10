import pickle
import numpy as np
from pprint import pprint


def open_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_internal_similarity(sense_embeddings: dict) -> float:
    """
    Computes the internal similarity of a set of sense embeddings.
    """
    sentences_count = len(sense_embeddings)
    total_similarity = 0.0

    for i in range(sentences_count):
        for j in range(sentences_count):
            if i != j:
                emb1 = sense_embeddings[i]["embedding"]
                emb2 = sense_embeddings[j]["embedding"]

                cosine_similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                total_similarity += cosine_similarity

    return total_similarity / (sentences_count**2 - sentences_count)


def compute_external_similarity(lemmapos_senses_embeddings: dict) -> float:
    """
    Computes the external similarity of a set of sense embeddings.
    """
    senses_similarities = []

    for sense in lemmapos_senses_embeddings.keys():
        sense_external_similarity = 0.0
        sense_embeddings = lemmapos_senses_embeddings[sense]

        other_senses = list(lemmapos_senses_embeddings.keys())
        other_senses.remove(sense)

        count_opposite = 0
        for sense_2 in other_senses:
            sense_embeddings_2 = lemmapos_senses_embeddings[sense_2]

            count_opposite += len(lemmapos_senses_embeddings[sense_2])

            for i in range(len(sense_embeddings)):
                for j in range(len(sense_embeddings_2)):
                    emb1 = sense_embeddings[i]["embedding"]
                    emb2 = sense_embeddings_2[j]["embedding"]

                    cosine_similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    sense_external_similarity += cosine_similarity

        sense_external_similarity /= count_opposite * len(sense_embeddings)
        senses_similarities.append(sense_external_similarity)

    return senses_similarities


embeddings_per_layer = open_pickle_file(
    "./embeddings/microsoft_deberta-v3-base/10/train_word_embeddings_per_layer_10.pkl"
)

print("Computing similarities of Deberta-v3-base with K = 5 ...\n")
for layer in embeddings_per_layer:
    embeddings = embeddings_per_layer[layer]

    sense_count = 0

    total_internal_similarity = 0.0
    total_external_similarity = []

    for lemmapos in embeddings:
        total_external_similarity.extend(
            compute_external_similarity(embeddings[lemmapos])
        )

        for lemma in embeddings[lemmapos]:
            sense_count += 1
            int_sim = compute_internal_similarity(embeddings[lemmapos][lemma])
            total_internal_similarity += int_sim

            # print(f"{lemmapos} - {lemma} - Internal Similarity: {int_sim}")

    print(
        f"Layer {layer} - Avg Internal Similarity: {round(total_internal_similarity/sense_count, 5)} - Avg External Similarity: {round(sum(total_external_similarity)/sense_count, 5)}"
    )
