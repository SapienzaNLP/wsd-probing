from nltk.corpus import wordnet as wn
from utils.wsd_utils import read_from_raganato, pos_map
from utils.wordnet_utils import (
    wn_offsets_from_lemmapos,
    synset_from_offset,
    wn_offset_from_sense_key,
    synsets_from_lemmapos,
)

from collections import Counter
import math
from pprint import pprint
import random

DEBUG = False

DATA_PATH = "./semcor_data/semcor.data.xml"
KEYS_PATH = "./semcor_data/semcor.gold.key.txt"


# This is a base function to read the SemCor corpus
def read_semcor(data_path: str, keys_path: str):
    offsets_counter = Counter()
    wsd_instances = []
    synset_list = []
    lemmas_list = []
    synset_list_unfiltered = []
    for _, _, wsd_sentence in read_from_raganato(data_path, keys_path):
        for wsd_instance in wsd_sentence:
            if wsd_instance.labels is None:
                continue
            else:
                wsd_instances.append(wsd_instance)
                labels = wsd_instance.labels
                for label in labels:
                    lemma = wn.lemma_from_key(label)
                    synset = lemma.synset()
                    synsets_list = wn.synsets(lemma.name())

                    synset_list_unfiltered.append(str(synset))

                    if DEBUG:
                        if str(synset) == "Synset('vacation.n.01')":
                            print(f"Key of vacation = {label}")
                            print(f"Synset of vacation = {synset}")
                            print(f"Lemma of vacation = {lemma}")
                            print(f"Word: {wsd_instance.annotated_token.text}")
                            print("\n")

                    if len(synsets_list) > 1:
                        # I want to track only polysemous words
                        synset_list.append(str(synset))
                        lemmas_list.append(lemma)

                wn_offsets = [
                    wn_offset_from_sense_key(_l) for _l in wsd_instance.labels
                ]
                offsets_counter.update(wn_offsets)

    lemmas_counter = Counter(lemmas_list)
    synset_counter = Counter(synset_list)
    synset_counter_unfiltered = Counter(synset_list_unfiltered)

    sorted_synsets = sorted(synset_counter.items(), key=lambda x: x[1], reverse=True)
    sorted_lemmas = sorted(lemmas_counter.items(), key=lambda x: x[1], reverse=True)
    sorted_synsets_unfiltered = sorted(
        synset_counter_unfiltered.items(), key=lambda x: x[1], reverse=True
    )

    with open("./synsets.txt", "w") as file:
        for synset, count in sorted_synsets:
            file.write(f"{synset}: {count}\n")

    with open("./lemmas.txt", "w") as file:
        for lemma, count in sorted_lemmas:
            file.write(f"{lemma}: {count}\n")

    with open("./synsets_unfiltered.txt", "w") as file:
        for synset, count in sorted_synsets_unfiltered:
            file.write(f"{synset}: {count}\n")

    return sorted_synsets, sorted_lemmas


def semcor_train_test_split(
    lemmas_grouped_per_synset_with_sentences: dict, TEST_SPLIT=0.2
):
    """
    Create a train/test split of the lemmas grouped per synset
    Args:
        lemmas_grouped_per_synset_with_sentences: dictionary containing as keys the synsets and as values a dictionary
        containing as keys the lemmas and as values a list of sentences
        TEST_SPLIT: percentage of examples to be used for testing

    Returns:
        Two dictionaries containing the train and test sets
    """
    train_set = {}
    test_set = {}
    for lemmapos in lemmas_grouped_per_synset_with_sentences.keys():
        lemmas = lemmas_grouped_per_synset_with_sentences[lemmapos]
        for lemma in lemmas:
            sentences = lemmas_grouped_per_synset_with_sentences[lemmapos][lemma]

            random.shuffle(sentences)

            if len(sentences) == 4:
                print("4 sentences found")
                test_examples_count = 1
            else:
                test_examples_count = math.floor(TEST_SPLIT * len(sentences))

            for i in range(test_examples_count):
                sentence = sentences.pop()
                if lemmapos not in test_set:
                    test_set[lemmapos] = {}
                if lemma not in test_set[lemmapos]:
                    test_set[lemmapos][lemma] = []

                test_set[lemmapos][lemma].append(sentence)

            for i in range(len(sentences)):
                sentence = sentences.pop()
                if lemmapos not in train_set:
                    train_set[lemmapos] = {}
                if lemma not in train_set[lemmapos]:
                    train_set[lemmapos][lemma] = []

                train_set[lemmapos][lemma].append(sentence)

    return train_set, test_set


## This function returns a dictionary containing as keys all the synsets present in the SemCor corpus
## and as values an object containing the frequency of the synset, the POS and the lemmas associated to it
## @param data_path: path to the SemCor corpus
## @param keys_path: path to the SemCor gold keys
def extract_all_synsets(data_path: str, keys_path: str) -> dict:
    synsets_dict = {}
    for _, _, wsd_sentence in read_from_raganato(data_path, keys_path):
        for wsd_instance in wsd_sentence:
            if wsd_instance.labels is None:
                continue
            else:
                # print(f"WSD Instance: {wsd_instance}")
                # print(f"Annotated token = {wsd_instance.annotated_token}")
                labels = wsd_instance.labels
                for label in labels:
                    # print(f"Label = {label}")
                    lemma = wn.lemma_from_key(label)
                    synset = lemma.synset()
                    synsets_list = synsets_from_lemmapos(
                        wsd_instance.annotated_token.text,
                        pos_map[wsd_instance.annotated_token.pos],
                    )
                    # print(f"Correct synset = {synset}")
                    # print(f"Synsets list = {synsets_list}")

                    # If word is polysemous
                    if len(synsets_list) > 1:
                        if synset in synsets_dict:
                            synsets_dict[synset]["frequency"] += 1
                            if (wsd_instance.annotated_token.lemma) not in synsets_dict[
                                synset
                            ]["lemma"]:
                                synsets_dict[synset]["lemma"].append(
                                    wsd_instance.annotated_token.lemma,
                                )
                        else:
                            synsets_dict[synset] = {
                                "frequency": 1,
                                "pos": wsd_instance.annotated_token.pos,
                                "lemma": [wsd_instance.annotated_token.lemma],
                            }

    # Sort the dictionary by frequency
    sorted_synset_dict = {
        k: v
        for k, v in sorted(
            synsets_dict.items(), key=lambda item: item[1]["frequency"], reverse=True
        )
    }

    with open("synsets_frequencies_with_POS.txt", "w") as file:
        # Write the dictionary content to the file
        for key, value in sorted_synset_dict.items():
            file.write(f"{key}: {value}\n")

    return sorted_synset_dict


def extract_all_lemmapos(data_path: str, keys_path: str) -> dict:
    """
    Read all SemCor corpus with raganato function and extract all lemmas with their POS and synsets
    Args:
        data_path: path to the SemCor corpus
        keys_path: path to the SemCor gold keys

    Returns:
        A dictionary containing as keys all the lemmas present in the SemCor corpus
        and as values an object containing the frequency of the lemma, the POS and the synsets associated to it
    """
    lemmas_dict = {}
    for _, _, wsd_sentence in read_from_raganato(data_path, keys_path):
        for wsd_instance in wsd_sentence:
            if wsd_instance.labels is None:
                continue
            else:
                # print(f"\n\nWSD Instance: {wsd_instance}")
                # print(f"Annotated token = {wsd_instance.annotated_token}")
                labels = wsd_instance.labels
                for label in labels:
                    # print(f"Label = {label}")
                    wordnet_lemma = wn.lemma_from_key(label)
                    # print(f"Wordnet lemma = {wordnet_lemma}")
                    lemma = wsd_instance.annotated_token.lemma
                    synset = wordnet_lemma.synset()

                    ## wordnet_lemma.name() as key?
                    if str(wordnet_lemma) not in lemmas_dict:
                        lemmas_dict[str(wordnet_lemma)] = {
                            "lemma": lemma,
                            "pos": wsd_instance.annotated_token.pos,
                            "synset": synset,
                        }

    with open("lemmapos_pairs.txt", "w") as file:
        # Write the dictionary content to the file
        for key, value in lemmas_dict.items():
            file.write(f"{key}: {value}\n")

    return lemmas_dict


def filter_and_group_senses_per_lemmapos(
    lemmas_with_more_than_k_occurrences: list,
    k: int,
) -> dict:
    senses_grouped_per_lemmapos = {}

    for lemma in lemmas_with_more_than_k_occurrences:
        wordnet_lemma = lemma
        lemma_name = lemmas_with_more_than_k_occurrences[lemma]["lemma"]
        pos = lemmas_with_more_than_k_occurrences[lemma]["pos"]

        entry = lemma_name + "#" + pos
        if entry not in senses_grouped_per_lemmapos:
            senses_grouped_per_lemmapos[entry] = []

        senses_grouped_per_lemmapos[entry].append(str(wordnet_lemma))

    senses_grouped_per_lemmapos_filtered = {}
    count_senses = 0
    for lemmapos in senses_grouped_per_lemmapos:
        if len(senses_grouped_per_lemmapos[lemmapos]) > 1:
            senses_grouped_per_lemmapos_filtered[lemmapos] = (
                senses_grouped_per_lemmapos[lemmapos]
            )
            count_senses += len(senses_grouped_per_lemmapos[lemmapos])

    with open(f"./word_data/senses_grouped_per_lemmapos_{k}.txt", "w") as file:
        # Write the dictionary content to the file
        for key, value in senses_grouped_per_lemmapos_filtered.items():
            file.write(f"{key}: {value}\n")

    return senses_grouped_per_lemmapos_filtered


## UNUSED
def extract_first_senses_synsets(synsets_dict: dict) -> list:
    res = []
    for syn in synsets_dict.keys():
        test_list = []
        for i in range(len(synsets_dict[syn]["lemma"])):
            synsets_list = synsets_from_lemmapos(
                synsets_dict[syn]["lemma"][i], pos_map[synsets_dict[syn]["pos"]]
            )
            res.append(synsets_list[0])
            test_list.append(synsets_list[0])

        if DEBUG:
            if len(list(dict.fromkeys(test_list))) != 1:
                print(
                    f"DIFFERENT SENSES: list={test_list}, lemmas = {synsets_dict[syn]['lemma']}\n"
                )

    # Removing duplicates before returning the result
    res = list(dict.fromkeys(res))

    with open("first_senses_synsets.txt", "w") as file:
        for syn in res:
            file.write(f"{syn}\n")
    file.close()

    return res


## This function returns a dictionary containing as keys all the lemmas present in the SemCor corpus
## and as values the frequencies of the lemmas. The result is SORTED on the frequencies
## @param data_path: path to the SemCor corpus
## @param keys_path: path to the SemCor gold keys
def extract_all_lemmas_frequencies(data_path: str, keys_path: str) -> dict:
    lemmas_dict = {}
    for _, _, wsd_sentence in read_from_raganato(data_path, keys_path):
        for wsd_instance in wsd_sentence:
            if wsd_instance.labels is None:
                continue
            else:
                labels = wsd_instance.labels
                for label in labels:
                    wordnet_lemma = wn.lemma_from_key(label)
                    lemma = wsd_instance.annotated_token.lemma
                    synset = wordnet_lemma.synset()

                    if str(wordnet_lemma) not in lemmas_dict:
                        lemmas_dict[str(wordnet_lemma)] = 1
                    else:
                        lemmas_dict[str(wordnet_lemma)] += 1

    lemmas_dict = {
        k: v
        for k, v in sorted(lemmas_dict.items(), key=lambda item: item[1], reverse=True)
    }

    with open("lemmas_frequencies.txt", "w") as file:
        # Write the dictionary content to the file
        for key, value in lemmas_dict.items():
            file.write(f"{key}: {value}\n")

    return lemmas_dict


def get_sentences_of_lemma(
    lemma: str, all_sentences: list, lemmas_filtered_grouped_per_word: dict
) -> list:
    res = []
    lemmas = lemmas_filtered_grouped_per_word[lemma]
    for lemma in lemmas:
        res.extend(all_sentences[lemma])
    return res


def main():
    lemmas_dict = extract_all_lemmas_frequencies(DATA_PATH, KEYS_PATH)


if __name__ == "__main__":
    main()
