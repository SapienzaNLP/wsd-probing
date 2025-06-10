import pickle
from nltk.corpus import wordnet as wn
import numpy as np
from pprint import pprint

from utils.embedding_utils import get_nearest_centroid
from utils.wordnet_utils import lemma_object_from_string


def read_pickle_dict(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save_dict_on_file(data, file_name) -> bool:
    """
    Save a dictionary on a file
    :param data: dictionary to save
    :param file_name: file name
    :return: True if the dictionary has been saved successfully, False otherwise
    """
    try:
        with open(file_name, "wb") as file:
            pickle.dump(data, file)
            return True
    except Exception as e:
        print("Error saving dictionary on file: " + str(e))
        return False


def vanilla_save_dict_on_file(data, file_name) -> bool:
    """
    Save a dictionary on a file
    :param data: dictionary to save
    :param file_name: file name
    :return: True if the dictionary has been saved successfully, False otherwise
    """
    try:
        with open(file_name, "w") as file:
            file.write(str(data))
            return True
    except Exception as e:
        print("Error saving dictionary on file: " + str(e))
        return False


def open_frequency_dict_from_file(file_name) -> dict:
    res = {}
    with open(file_name, "r") as file:
        for line in file:
            key, value = line.strip().split(": ")
            res[key] = int(value)
    return res


def save_all_words_on_file(all_words: dict, k: int) -> bool:
    with open(f"./word_data/all_words_{k}.txt", "w") as file:
        file.write(
            f"Total number of words: {sum(len(words) for words in all_words.values())}\n"
        )
        file.write(f"Total number of lemmas: {len(all_words)}\n\n")
        for lemma in all_words.keys():
            file.write(f"{lemma}: {all_words[lemma]}\n")


def group_word_embeddings_per_layer(word_embeddings: dict, LAYERS_LIST) -> dict:
    """
    Function that group the word embeddings per layer

    Args:
        word_embeddings: dictionary of the form {"lemmapos": {"lemma": [{"sentence": sentence, "target_word_position": target_word_position, "embedding": embedding}]}}
    Returns:
        A dictionary of the form {"layer": {"lemmapos": {"lemma": [sentence, embedding]}}}
    """
    res = {
        k: {
            lemmapos: {lemma: [] for lemma in word_embeddings[lemmapos].keys()}
            for lemmapos in word_embeddings.keys()
        }
        for k in LAYERS_LIST
    }

    for lemmapos in word_embeddings.keys():
        for lemma in word_embeddings[lemmapos].keys():
            sentences = word_embeddings[lemmapos][lemma]
            for sentence in sentences:
                for layer in LAYERS_LIST:
                    res[layer][lemmapos][lemma].append(
                        {
                            "sentence": sentence["sentence"],
                            "embedding": sentence["embedding"][layer]["embedding"],
                        }
                    )

    return res


def compute_errors(test_word_embeddings: dict, centroids: dict, DISTANCE_FUNCTION):
    errors = {}
    corrects = {}

    for lemmapos in test_word_embeddings.keys():
        lemmas_list = test_word_embeddings[lemmapos]

        for lemma in lemmas_list:
            errors[str(lemma)] = {}

            for word_embedding in lemmas_list[str(lemma)]:
                nearest_centroid_elem = get_nearest_centroid(
                    word_embedding["embedding"],
                    [
                        {
                            "centroid": centroids[str(list_lemma)],
                            "lemma": str(list_lemma),
                        }
                        for list_lemma in lemmas_list
                    ],
                    DISTANCE_FUNCTION=DISTANCE_FUNCTION,
                )
                nearest_centroid = nearest_centroid_elem["nearest_centroid"]

                if str(nearest_centroid) != str(lemma):
                    if str(nearest_centroid) in errors[str(lemma)]:
                        errors[str(lemma)][str(nearest_centroid)]["sentences"].append(
                            word_embedding["sentence"]
                        )
                        errors[str(lemma)][str(nearest_centroid)]["distances"].append(
                            nearest_centroid_elem["distance"]
                        )
                        errors[str(lemma)][str(nearest_centroid)]["count"] += 1
                    else:
                        errors[str(lemma)][str(nearest_centroid)] = {
                            "sentences": [word_embedding["sentence"]],
                            "distances": [nearest_centroid_elem["distance"]],
                            "count": 1,
                        }
                else:
                    ## UNNECESSARY RN
                    if str(nearest_centroid) in corrects:
                        corrects[str(nearest_centroid)]["sentences"].append(
                            word_embedding["sentence"]
                        )
                        corrects[str(nearest_centroid)]["distances"].append(
                            nearest_centroid_elem["distance"]
                        )
                        corrects[str(nearest_centroid)]["count"] += 1
                    else:
                        corrects[str(nearest_centroid)] = {
                            "sentences": [word_embedding["sentence"]],
                            "distances": [nearest_centroid_elem["distance"]],
                            "count": 1,
                        }

    return errors, corrects


def explain_errors_on_file(
    errors: dict,
    corrects: dict,
    test_word_embeddings: dict,
    FILE_PATH: str,
    DISTANCE_FUNCTION: str,
    k: int,
) -> float:
    count_wrong_closer_than_corrects = 0

    total_number_of_sentences = 0
    for lemmapos in test_word_embeddings.keys():
        for lemma in test_word_embeddings[lemmapos].keys():
            total_number_of_sentences += len(test_word_embeddings[lemmapos][lemma])

    test_lemmas_occurences = {}
    for lemmapos in test_word_embeddings.keys():
        for lemma in test_word_embeddings[lemmapos].keys():
            if lemma in test_lemmas_occurences:
                test_lemmas_occurences[lemma] += len(
                    test_word_embeddings[lemmapos][lemma]
                )
            else:
                test_lemmas_occurences[lemma] = len(
                    test_word_embeddings[lemmapos][lemma]
                )

    with open(FILE_PATH, "w") as file:
        file.write(f"Errors for k = {k}\n")
        file.write(f"Distance function: {DISTANCE_FUNCTION}\n\n")
        total_number_of_errors = 0

        for lemma in errors.keys():
            for wrong_lemma in errors[lemma].keys():
                total_number_of_errors += errors[lemma][wrong_lemma]["count"]

        file.write(f"Number of total errors = {total_number_of_errors}\n")
        file.write(f"Number of total test sentences = {total_number_of_sentences}\n")
        total_accuracy = 1 - total_number_of_errors / total_number_of_sentences
        file.write(f"Total Accuracy of BERT = {total_accuracy}\n\n")

        wrong_lemmas_count = 0

        for lemma in errors.keys():
            file.write(f"Errors for {lemma}\n")
            file.write(
                f"Sense definition: {lemma_object_from_string(lemma).synset().definition()}\n"
            )

            sentences_of_lemma = test_lemmas_occurences[lemma]

            file.write(f"Number of sentences: {sentences_of_lemma}\n")

            error_for_this_lemma = 0
            for wrong_lemma in errors[lemma].keys():
                error_for_this_lemma += errors[lemma][wrong_lemma]["count"]

            file.write(f"Accuracy: {1 - error_for_this_lemma/sentences_of_lemma}\n\n")

            for wrong_lemma in errors[lemma].keys():
                wrong_lemmas_count += 1
                file.write(
                    f"There are {errors[lemma][wrong_lemma]['count']} embedding{'s' if len(errors[lemma]) > 1 else ''} closer to {wrong_lemma} ({lemma_object_from_string(wrong_lemma).synset().definition()})\n"
                )
                avg_distance_of_wrong_embeddings = np.mean(
                    errors[lemma][wrong_lemma]["distances"]
                )
                file.write(
                    f"Average distance of WRONG embeddings from nearest centroid: {avg_distance_of_wrong_embeddings}\n"
                )
                avg_distance_of_correct_embeddings = 1e10
                if lemma in corrects.keys():
                    avg_distance_of_correct_embeddings = np.mean(
                        corrects[lemma]["distances"]
                    )
                    file.write(
                        f"Average distance of CORRECT embeddings: {avg_distance_of_correct_embeddings}\n"
                    )
                else:
                    file.write(f"No correct embeddings for this lemma\n")

                if (
                    avg_distance_of_wrong_embeddings
                    < avg_distance_of_correct_embeddings
                ):
                    count_wrong_closer_than_corrects += 1
                    file.write(
                        "AVG DISTANCE OF WRONG EMBEDDINGS IS SMALLER THAN AVG DISTANCE OF CORRECT EMBEDDINGS\n"
                    )

                file.write("\t\t|\n")
                for i in range(min(20, len(errors[lemma][wrong_lemma]["sentences"]))):
                    file.write("\t\t|\n")
                    file.write(
                        f"\t\t{i+1} - Dist: {errors[lemma][wrong_lemma]['distances'][i]} {' '.join(errors[lemma][wrong_lemma]['sentences'][i])}\n"
                    )
                file.write("\n")

            file.write("\n-----------------------------------------\n")

        file.write(
            f"Number of wrong embeddings closer to wrong centroid than to correct centroid: {count_wrong_closer_than_corrects}\n"
        )
        file.write(
            f"Percentage of wrong embeddings closer than correct embeddings to wrong centroid: {count_wrong_closer_than_corrects/wrong_lemmas_count}\n"
        )
        file.write(f"Wrong lemmas count: {wrong_lemmas_count}\n")
    file.close()
    return total_accuracy
