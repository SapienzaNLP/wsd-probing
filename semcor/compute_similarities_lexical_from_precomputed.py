import pickle
import numpy as np
from pprint import pprint
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
BEST_LAYER = 7
EMBEDDINGS = "train_word_embeddings_per_layer_5"
EXTERNAL_SIM_FILE = f"./embeddings/external_similarities_{EMBEDDINGS}.txt"
INTERNAL_SIM_FILE = f"./embeddings/internal_similarities_{EMBEDDINGS}.txt"


def open_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_internal_similarity_lexical(sense_sentences: dict) -> float:
    """
    Computes the internal similarity of a set of sense embeddings.
    """
    sentences_count = len(sense_sentences)
    total_similarity = 0.0

    for i in range(sentences_count):
        for j in range(sentences_count):
            if i != j:
                sent1 = sense_sentences[i]["sentence"]
                sent2 = sense_sentences[j]["sentence"]

                ## compute lexical similarity with Jaccard
                sent1 = set(sent1)
                sent2 = set(sent2)

                sent1 = set([word for word in sent1 if word not in stop_words])
                sent2 = set([word for word in sent2 if word not in stop_words])

                intersection = len(sent1.intersection(sent2))
                union = len(sent1.union(sent2))

                jaccard_similarity = float(intersection) / float(union)

                total_similarity += jaccard_similarity

    return total_similarity / (sentences_count**2 - sentences_count)


def compute_external_similarity_lexical(lemmapos_senses_sentences: dict) -> float:
    external_similarities = []

    normalization_factor = [len(sense) for sense in lemmapos_senses_sentences.values()]
    normalization_factor = np.prod(normalization_factor)

    for sense in lemmapos_senses_sentences:
        sense_external_similarity = 0.0
        other_senses = list(lemmapos_senses_sentences.keys())
        other_senses.remove(sense)

        pbar_sense = tqdm(lemmapos_senses_sentences[sense])
        for elem in pbar_sense:
            for other_sense in other_senses:
                pbar_sense.set_description(
                    f"External Similarity: {sense} - {other_sense}"
                )
                for other_elem in lemmapos_senses_sentences[other_sense]:
                    sent1 = set(elem["sentence"])
                    sent2 = set(other_elem["sentence"])

                    sent1 = set([word for word in sent1 if word not in stop_words])
                    sent2 = set([word for word in sent2 if word not in stop_words])

                    intersection = len(sent1.intersection(sent2))
                    union = len(sent1.union(sent2))

                    jaccard_similarity = float(intersection) / float(union)

                    sense_external_similarity += jaccard_similarity

        sense_external_similarity /= normalization_factor
        external_similarities.append(sense_external_similarity)

        with open(EXTERNAL_SIM_FILE, "a") as f:
            f.write(f"{sense}: {round(jaccard_similarity, 10)}\n")

    return external_similarities


def main():
    train_embeddings = open_pickle_file(
        f"../ALL/data/microsoft_deberta-v3-base_{EMBEDDINGS}.pkl"
    )
    train_embeddings = train_embeddings[BEST_LAYER]

    sense_count = 0
    external_similarities_list = []
    internal_similarities_list = []

    pbar = tqdm(train_embeddings)
    for lemmapos in pbar:
        pbar.set_description(f"Processing {lemmapos}")

        with open(EXTERNAL_SIM_FILE, "a") as f:
            f.write(f"{'-'*50}\n{lemmapos}\n")

        with open(INTERNAL_SIM_FILE, "a") as f:
            f.write(f"{'-'*50}\n{lemmapos}\n")

        external_similarities = compute_external_similarity_lexical(
            train_embeddings[lemmapos]
        )
        external_similarities_list.extend(external_similarities)

        for sense in train_embeddings[lemmapos]:
            sense_count += 1
            internal_similarity = compute_internal_similarity_lexical(
                train_embeddings[lemmapos][sense]
            )
            with open(INTERNAL_SIM_FILE, "a") as f:
                f.write(f"{sense}: {round(internal_similarity, 10)}\n")
            internal_similarities_list.append(internal_similarity)

    print(
        f"Avg Internal Similarity (lexical): {round(sum(internal_similarities_list)/sense_count, 10)}"
    )
    with open(EXTERNAL_SIM_FILE, "a") as f:
        f.write(f"{'-'*50}\n")
        f.write(
            f"Avg External Similarity (lexical): {round(sum(external_similarities_list)/sense_count, 10)}"
        )

    print(
        f"Avg External Similarity (lexical): {round(sum(external_similarities_list)/sense_count, 10)}"
    )
    with open(INTERNAL_SIM_FILE, "a") as f:
        f.write(f"{'-'*50}\n")
        f.write(
            f"Avg Internal Similarity (lexical): {round(sum(internal_similarities_list)/sense_count, 10)}"
        )
    return 0


if __name__ == "__main__":
    main()
