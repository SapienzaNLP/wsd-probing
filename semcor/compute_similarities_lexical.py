import pickle
import numpy as np
from pprint import pprint
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

from utils.wsd_utils import (
    expand_raganato_path,
    read_from_raganato,
    pos_map,
    WSDInstance,
)
from utils.wordnet_utils import (
    wn_offsets_from_lemmapos,
    synset_from_offset,
    wn_offset_from_sense_key,
    synsets_from_lemmapos,
    lemma_object_from_string,
    get_sense_position,
)

from utils.embedding_utils import (
    extract_target_sentences,
    compute_embedding,
    compute_corpus_embeddings,
    extract_target_sentences_from_synsets_list,
    extract_target_sentences_from_lemmas_list,
    add_sentences_for_each_lemma,
    compute_corpus_centroids,
    get_nearest_centroid,
)

from utils.visualization_utils import plot_word_embeddings, plot_heatmap

from utils.semcor_utils import (
    extract_all_synsets,
    extract_all_lemmapos,
    get_sentences_of_lemma,
    filter_and_group_senses_per_lemmapos,
    semcor_train_test_split,
)
from utils.general_utils import (
    pickle_save_dict_on_file,
    vanilla_save_dict_on_file,
    save_all_words_on_file,
    open_frequency_dict_from_file,
    group_word_embeddings_per_layer,
    compute_errors,
    explain_errors_on_file,
)


EXTRACT_SENTENCES = True
DATA_PATH = "./semcor_data/semcor.data.xml"
KEYS_PATH = "./semcor_data/semcor.gold.key.txt"
DEBUG_EXACT_K = True
k = 5
stop_words = set(stopwords.words("english"))


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

                # sent1 = set([word for word in sent1 if word not in stop_words])
                # sent2 = set([word for word in sent2 if word not in stop_words])

                intersection = len(sent1.intersection(sent2))
                union = len(sent1.union(sent2))

                jaccard_similarity = intersection / union

                total_similarity += jaccard_similarity

    return total_similarity / (sentences_count**2 - sentences_count)


def compute_external_similarity_lexical(lemmapos_senses_sentences: dict) -> float:
    """
    Computes the external similarity of a set of sense sentences.
    """
    senses_similarities = []

    for sense in lemmapos_senses_sentences.keys():
        sense_external_similarity = 0.0
        sense_sentences = lemmapos_senses_sentences[sense]

        other_senses = list(lemmapos_senses_sentences.keys())
        other_senses.remove(sense)

        count_opposite = 0
        for sense_2 in other_senses:
            sense_sentences_2 = lemmapos_senses_sentences[sense_2]

            count_opposite += len(lemmapos_senses_sentences[sense_2])

            for i in range(len(sense_sentences)):
                for j in range(len(sense_sentences_2)):
                    sent1 = sense_sentences[i]["sentence"]
                    sent2 = sense_sentences_2[j]["sentence"]

                    ## compute lexical similarity with Jaccard
                    sent1 = set(sent1)
                    sent2 = set(sent2)
                    intersection = len(sent1.intersection(sent2))
                    union = len(sent1.union(sent2))

                    jaccard_similarity = intersection / union

                    sense_external_similarity += jaccard_similarity

        sense_external_similarity /= count_opposite * len(sense_sentences)
        senses_similarities.append(sense_external_similarity)

    return senses_similarities


## This function return a list of lemmas that appear more than k times in Semcor
def get_lemmas_with_at_least_k_occurrences(
    all_lemmapos_dict: dict, lemmas_frequencies: dict, k: int
) -> list:
    res = {}
    for sense in all_lemmapos_dict:
        if lemmas_frequencies[sense] >= k:
            res[sense] = all_lemmapos_dict[sense]
    return res


lemmas_frequencies = open_frequency_dict_from_file("./word_data/lemmas_frequencies.txt")

print(f"Len lemmas_frequencies: {len(lemmas_frequencies)}")

## all_lemmapos_dict is a dictionary that contains as keys the Lemma object of wordnet
## and as values a DICT that contains the LEMMA NAME, the POS tag and the SYNSET
all_lemmapos_dict = extract_all_lemmapos(DATA_PATH, KEYS_PATH)

print(f"Len all_lemmapos_dict: {len(all_lemmapos_dict)}")

print("K = ", k)
print(f"Debug exact k: {DEBUG_EXACT_K}")
lemmas_with_more_than_k_occurrences = get_lemmas_with_at_least_k_occurrences(
    all_lemmapos_dict, lemmas_frequencies, k
)

print(
    f"Len lemmas_with_more_than_k_occurrences: {len(lemmas_with_more_than_k_occurrences)}"
)

## Dictionary containing as keys the first sense of the lemmas and as values
## the list of lemmas that have more than k occurrences in Semcor under that synset
## This function also FILTER OUT the MONOSEMOUS lemmas
lemmas_grouped_per_lemmapos_filtered = filter_and_group_senses_per_lemmapos(
    lemmas_with_more_than_k_occurrences, k
)

target_lemmas = []
for lemmapos in lemmas_grouped_per_lemmapos_filtered.keys():
    for lemma in lemmas_grouped_per_lemmapos_filtered[lemmapos]:
        target_lemmas.append(lemma)

all_sentences = extract_target_sentences_from_lemmas_list(target_lemmas)

lemmas_grouped_per_lemmapos_with_sentences = add_sentences_for_each_lemma(
    lemmas_grouped_per_lemmapos_filtered,
    all_sentences,
    k,
    DEBUG_EXACT_K,
)

## CREATE TRAIN AND TEST SPLITS
train_set, test_set = semcor_train_test_split(
    lemmas_grouped_per_lemmapos_with_sentences
)

train_embeddings = open_pickle_file(
    "../ALL/data/microsoft_deberta-v3-base_train_word_embeddings_per_layer_5.pkl"
)


for lemmapos in tqdm(train_set):
    sense_count = 0

    total_internal_similarity = 0.0
    total_external_similarity = []

    external_similarities = compute_external_similarity_lexical(train_set[lemmapos])
    total_external_similarity.extend(external_similarities)

    for lemma in train_set[lemmapos]:
        sense_count += 1
        int_sim = compute_internal_similarity_lexical(train_set[lemmapos][lemma])
        total_internal_similarity += int_sim

    # print(f"{lemmapos} - {lemma} - Internal Similarity: {int_sim}")

print(
    f"Avg Internal Similarity (lexical): {round(total_internal_similarity/sense_count, 5)} - Avg External Similarity (lexical): {round(sum(total_external_similarity)/sense_count, 5)}"
)
