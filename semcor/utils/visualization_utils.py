import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from pprint import pprint
from utils.embedding_utils import (
    extract_target_sentences_from_lemmas_list,
)
from utils.embedding_utils import (
    compute_embedding,
)


def plot_word_embeddings(
    word_embeddings, word_embeddings_2, target_synset, target_synset_2
):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(word_embeddings)
    pca_result2 = pca.fit_transform(word_embeddings_2)

    plt.scatter(
        pca_result[:, 0], pca_result[:, 1], label=target_synset, alpha=0.5, c="blue"
    )

    plt.scatter(
        pca_result2[:, 0], pca_result2[:, 1], label=target_synset_2, alpha=0.5, c="red"
    )

    # The centroid is always at [0,0] ???
    # centroid_1 = np.mean(pca_result, axis=0)
    # centroid_2 = np.mean(pca_result2, axis=0)

    # plt.scatter(centroid_1[0], centroid_1[1], marker="8",
    #             s=150, c='blue', label=f"Centroid of {target_synset}")
    # plt.scatter(centroid_2[0], centroid_2[1], marker="x",
    #             s=150,  c='red', label=f"Centroid of {target_synset_2}")

    plt.title("Word Embeddings Visualization with PCA")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend()
    plt.show()


# 28*28 = 784
def plot_heatmap(word_embeddings):
    num_plots = len(word_embeddings)
    # Create subplots with 2 rows and 5 columns
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

    for i, ax in enumerate(axes.flatten()):
        if i < num_plots:
            data_list = word_embeddings[i].tolist()
            data_list = data_list + [0] * (784 - len(data_list))
            elements_per_row = 28
            num_rows = len(data_list) // elements_per_row
            rows = [
                data_list[j * elements_per_row : (j + 1) * elements_per_row]
                for j in range(num_rows)
            ]
            rows = [[round(value) for value in row] for row in rows]
            np_data = pd.DataFrame(rows)
            sns.heatmap(np_data, cmap="YlGnBu", ax=ax, vmax=3.0, vmin=-3.0)
            ax.set_title(f"Plot {i+1}")

            # Remove x and y labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()


def umap_scatterplot(word_embeddings):
    reducer = umap.UMAP(n_neighbors=5, n_components=2, metric="cosine")
    embedding = reducer.fit_transform(word_embeddings)

    plt.scatter(
        embedding[:, 0], embedding[:, 1], label="label", alpha=0.5, c="blue", marker="."
    )

    plt.title("Word Embeddings Visualization with UMAP")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend()
    plt.show()


def main():
    all_sentences = extract_target_sentences_from_lemmas_list(
        [
            "Lemma('die.v.01.die')",
            "Lemma('die.n.01.dice')",
            "Lemma('die.n.01.die')",
            "Lemma('die.v.01.perish')",
        ]
    )

    word_embeddings = {}
    for i, lemma in enumerate(all_sentences.keys()):
        print(f"Computing embeddings for lemma {i+1}: ", lemma, "...")
        word_embeddings[lemma] = [
            compute_embedding(
                sentence["sentence"],
                sentence["target_word_position"],
                is_split_into_words=False,
            )
            for sentence in all_sentences[lemma]
        ]

    word_embeddings_list = []
    for lemma in word_embeddings.keys():
        for elem in word_embeddings[lemma]:
            word_embeddings_list.append(elem["embedding"])

    umap_scatterplot(word_embeddings_list)
    return 0


if __name__ == "__main__":
    main()
