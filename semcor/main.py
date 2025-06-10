import time
from nltk.corpus import wordnet as wn
from collections import Counter
import numpy as np
import sys
import os
import tqdm
import copy
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd


from utils.wsd_utils import (
    pos_map,
)
from utils.wordnet_utils import (

    synsets_from_lemmapos,
)

from utils.embedding_utils import (
    compute_corpus_embeddings,
    extract_target_sentences_from_lemmas_list,
    add_sentences_for_each_lemma,
    compute_corpus_centroids,
)

from utils.semcor_utils import (
    extract_all_lemmapos,
    filter_and_group_senses_per_lemmapos,
    semcor_train_test_split,
)
from utils.general_utils import (
    pickle_save_dict_on_file,
    open_frequency_dict_from_file,
    group_word_embeddings_per_layer,
    compute_errors,
    explain_errors_on_file,
)

DATA_PATH = "./semcor_data/semcor.data.xml"
KEYS_PATH = "./semcor_data/semcor.gold.key.txt"

DEBUG = False

BANNED_LEMMAS = []

SAVE_EMBEDDINGS = True

SAVE_FILE = False


def get_synset_list_from_lemmapos(
    all_lemmapos_dict: dict, syn_frequencies: dict, k: int
) -> None:
    analyzed = []
    res = []
    for lemmapos in all_lemmapos_dict:
        if {
            "lemma": all_lemmapos_dict[lemmapos]["lemma"],
            "pos": all_lemmapos_dict[lemmapos]["pos"],
        } not in analyzed:
            analyzed.append(
                {
                    "lemma": all_lemmapos_dict[lemmapos]["lemma"],
                    "pos": all_lemmapos_dict[lemmapos]["pos"],
                }
            )
            synsets_list = synsets_from_lemmapos(
                all_lemmapos_dict[lemmapos]["lemma"],
                pos_map[all_lemmapos_dict[lemmapos]["pos"]],
            )

            matching_synsets = []

            if len(synsets_list) > 1:
                if (
                    str(synsets_list[0]) in syn_frequencies
                    and syn_frequencies[str(synsets_list[0])] > k
                ):
                    # print(f"{synsets_list[0]}: {syn_frequencies[str(synsets_list[0])]}")
                    matching_synsets.append(synsets_list[0])
                    for synset in synsets_list[1:]:
                        if (
                            str(synset) in syn_frequencies
                            and syn_frequencies[str(synset)] > k
                        ):
                            # print(f"{synset}: {syn_frequencies[str(synset)]}")
                            matching_synsets.append(synset)

            if len(matching_synsets) > 1:
                # print("\n-----------------------------------------\n")
                # print(
                #     f"Lemma: {all_lemmapos_dict[lemmapos]['lemma']}, Wordnet lemma entry: {lemmapos}\n"
                # )
                # for i, elem in enumerate(matching_synsets):
                #     print(f"{i+1}. {elem} : {syn_frequencies[str(elem)]}\n")

                res.append({"WordNet Lemma": lemmapos, "Synsets": matching_synsets})

    return res


def extract_lemma_name(lemma_string):
    """
    Function that extract the lemma name from a string of the form "Lemma('lemma_name.pos.01.synset_name')"
    Args:
        lemma_string: string of the form "Lemma('lemma_name.pos.01.synset_name')"
    Returns:
        The lemma name
    """
    lemma_parts = lemma_string.split(".")
    lemma_name = lemma_parts[-1]
    lemma_name = lemma_name.rstrip("')")
    return lemma_name


## This function return a list of lemmas that appear more than k times in Semcor
def get_lemmas_with_at_least_k_occurrences(
    all_lemmapos_dict: dict, lemmas_frequencies: dict, k: int
) -> list:
    res = {}
    for sense in all_lemmapos_dict:
        if lemmas_frequencies[sense] >= k:
            res[sense] = all_lemmapos_dict[sense]
    return res


def main(
    k_list,
    DISTANCE_FUNCTION,
    TRANSFORMER_MODEL,
    LAYERS_LIST,
    SAVING_DIRECTORY,
    DEBUG_EXACT_K,
):
    syn_frequencies = open_frequency_dict_from_file(
        "./word_data/synsets_frequencies.txt"
    )
    lemmas_frequencies = open_frequency_dict_from_file(
        "./word_data/lemmas_frequencies.txt"
    )

    print(f"Len lemmas_frequencies: {len(lemmas_frequencies)}")

    ## all_lemmapos_dict is a dictionary that contains as keys the Lemma object of wordnet
    ## and as values a DICT that contains the LEMMA NAME, the POS tag and the SYNSET
    all_lemmapos_dict = extract_all_lemmapos(DATA_PATH, KEYS_PATH)

    print(f"Len all_lemmapos_dict: {len(all_lemmapos_dict)}")

    for k in k_list:
        print("K = ", k)
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

        all_sentences_assert = copy.deepcopy(all_sentences)

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

        for synset in train_set.keys():
            for lemma in train_set[synset].keys():
                if DEBUG_EXACT_K:
                    assert (
                        len(train_set[synset][lemma]) + len(test_set[synset][lemma])
                        == k
                    )
                else:
                    assert len(train_set[synset][lemma]) + len(
                        test_set[synset][lemma]
                    ) == len(all_sentences_assert[lemma])

        print("Train-Test splits created")
        del all_sentences_assert

        ## COMPUTE TRAIN AND TEST EMBEDDINGS
        train_word_embeddings = compute_corpus_embeddings(
            train_set, TRANSFORMER_MODEL=TRANSFORMER_MODEL, LAYERS_LIST=LAYERS_LIST
        )

        test_word_embeddings = compute_corpus_embeddings(
            test_set, TRANSFORMER_MODEL=TRANSFORMER_MODEL, LAYERS_LIST=LAYERS_LIST
        )

        train_word_embeddings_per_layer = group_word_embeddings_per_layer(
            train_word_embeddings, LAYERS_LIST
        )

        test_word_embeddings_per_layer = group_word_embeddings_per_layer(
            test_word_embeddings, LAYERS_LIST
        )

        ## save train and test embeddings with pickle
        if SAVE_EMBEDDINGS:
            try:
                if DEBUG_EXACT_K:
                    pickle_save_dict_on_file(
                        train_word_embeddings_per_layer,
                        f"{SAVING_DIRECTORY}_train_word_embeddings_per_layer_exact_k_{k}.pkl",
                    )
                    pickle_save_dict_on_file(
                        test_word_embeddings_per_layer,
                        f"{SAVING_DIRECTORY}_test_word_embeddings_per_layer_exact_k_{k}.pkl",
                    )
                else:
                    pickle_save_dict_on_file(
                        train_word_embeddings_per_layer,
                        f"{SAVING_DIRECTORY}_train_word_embeddings_per_layer_{k}.pkl",
                    )
                    pickle_save_dict_on_file(
                        test_word_embeddings_per_layer,
                        f"{SAVING_DIRECTORY}_test_word_embeddings_per_layer_{k}.pkl",
                    )

                print("Embeddings saved correctly")
            except:
                print("Error while saving embeddings")
                return 0

        for layer in LAYERS_LIST:
            train_word_embeddings = train_word_embeddings_per_layer[layer]
            test_word_embeddings = test_word_embeddings_per_layer[layer]

            centroids, centroids_variances, centroids_std = compute_corpus_centroids(
                train_word_embeddings
            )

            # ------------------------------#
            #       RELEASE SOME MEMORY     #
            # ------------------------------#
            del train_word_embeddings

            errors, corrects = compute_errors(
                test_word_embeddings, centroids, DISTANCE_FUNCTION
            )

            if not os.path.exists(f"./errors"):
                os.makedirs(f"./errors")

            if not os.path.exists(f"./errors/{k}"):
                os.makedirs(f"./errors/{k}")

            if not os.path.exists(f"./errors/{k}/{SAVING_DIRECTORY}"):
                os.makedirs(f"./errors/{k}/{SAVING_DIRECTORY}")

            if not os.path.exists(
                f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}"
            ):
                os.makedirs(f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}")

            if DEBUG_EXACT_K:
                if not os.path.exists(
                    f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}/exact_K"
                ):
                    os.makedirs(
                        f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}/exact_K"
                    )

            if DEBUG_EXACT_K:
                FILE_PATH = f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}/exact_K/{SAVING_DIRECTORY}_errors_{DISTANCE_FUNCTION}_layer={layer}.txt"

            else:
                FILE_PATH = f"./errors/{k}/{SAVING_DIRECTORY}/{DISTANCE_FUNCTION}/{SAVING_DIRECTORY}_errors_{DISTANCE_FUNCTION}_layer={layer}.txt"

            model_accuracy = explain_errors_on_file(
                errors, corrects, test_word_embeddings, FILE_PATH, DISTANCE_FUNCTION, k
            )

            print(f"LAYER {layer} => Model accuracy: {model_accuracy}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    k_list = [int(arg) for arg in args]

    TRANSFORMER_MODEL_LIST = [
        # "bert-base-uncased",
        # "bert-base-cased",
        # "google/electra-base-discriminator",
        # "roberta-base",
        "microsoft/deberta-v3-base",
    ]
    LAYERS_LIST = [i for i in range(13)]

    DEBUG_EXACT_K = False

    if len(k_list) > 0:
        print("K list: ", k_list)
        print(f"Debug exact k: {DEBUG_EXACT_K}")
        print("TRANSFORMER MODEL LIST: ", TRANSFORMER_MODEL_LIST)
        for TRANSFORMER_MODEL in TRANSFORMER_MODEL_LIST:
            print("ANALYZING TRANSFORMER MODEL: ", TRANSFORMER_MODEL)
            SAVING_DIRECTORY = TRANSFORMER_MODEL.replace("/", "_")
            main(
                k_list,
                DISTANCE_FUNCTION="cosine",
                TRANSFORMER_MODEL=TRANSFORMER_MODEL,
                LAYERS_LIST=LAYERS_LIST,
                SAVING_DIRECTORY=SAVING_DIRECTORY,
                DEBUG_EXACT_K=DEBUG_EXACT_K,
            )
    else:
        print("Please specify a k value.")
