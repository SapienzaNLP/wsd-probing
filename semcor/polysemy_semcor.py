from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import List, Dict
from utils.wsd_utils import expand_raganato_path, read_from_raganato
import csv
from pprint import pprint

import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import wordnet as wn


pos_classes = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "PRT": "r"}

word_inventory = {"en": {}}
for word in wn.words():
    word_inventory["en"][word] = {}
    for pos in pos_classes.values():
        word_inventory["en"][word][pos] = []
        for synset in wn.synsets(word, pos=pos):
            synset_id = synset.name()
            definition = synset.definition()
            word_inventory["en"][word][pos].append(synset_id)


def load_raganato(dataset_path: Path, dataset_name: str, convert_bn: bool = True):
    total_ambiguity = 0
    counter = 0

    pprint(word_inventory)

    for idx, sentence_data in tqdm(
        enumerate(
            read_from_raganato(
                *expand_raganato_path(str(dataset_path / dataset_path.name))
            )
        ),
        desc=dataset_path.name,
    ):
        doc_id, sentence_id, sentence = sentence_data

        instances = []

        for idx, word in enumerate(sentence):
            if word.instance_id != None:
                instances.append(
                    {
                        "word": word.annotated_token.text,
                        "lemma": word.annotated_token.lemma,
                        "pos": word.annotated_token.pos,
                        "label": word.labels,
                        "position": idx,
                        "id": word.instance_id,
                    }
                )

        for instance in instances:
            entry = {}
            mapped_pos = pos_classes[instance["pos"]]
            entry["candidates"] = word_inventory["en"][instance["lemma"]][mapped_pos]

            total_ambiguity += len(entry["candidates"])
            counter += 1

    # compute the avg. polisemy degree
    print(f"Avg. ambiguity degree SemCor = {total_ambiguity/counter}")

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="semcor")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name
    raganato_path = Path(f"semcor_data/{dataset_name}")

    load_raganato(raganato_path, dataset_name)
