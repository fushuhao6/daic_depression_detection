import copy
import os
import json
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cosine, cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def embed_paragraph(model, tokenizer, paragraph):
    if tokenizer is None:
        # Example paragraph
        paragraph = "Your paragraph goes here."

        # Get the embedding
        paragraph_embedding = model.encode(paragraph)
    else:
        # Tokenize the input paragraph
        inputs = tokenizer(paragraph, return_tensors='pt', max_length=512, truncation=True)

        # Get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the embeddings of the [CLS] token for the paragraph representation
        paragraph_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    return paragraph_embedding


def embed_data(model, tokenizer, data):
    results = []
    for d in data:
        embedding = embed_paragraph(model, tokenizer, d)
        results.append(embedding)

    results = np.vstack(results)
    return results

def read_data_from_json_file(json_file, text_tag=('Synopsis', 'Sentiment')):
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []
    scores = []
    pids = []
    for d in data:
        text = [d[tag] for tag in text_tag]
        text = ' '.join(text)
        results.append(text)
        scores.append(d['PHQ8_Score'])
        participant_tag = 'Original_Participant_ID' if 'Original_Participant_ID' in d else 'Participant_ID'
        pids.append(d[participant_tag])
    return results, scores, pids


# Function to calculate average and minimum distance, excluding self-distances
def calculate_distances(embeddings, metric='euclidean'):
    pairwise_distances = squareform(pdist(embeddings, metric))
    np.fill_diagonal(pairwise_distances, np.inf)  # Exclude self-distances by setting them to infinity
    avg_distance = np.mean(pairwise_distances[np.isfinite(pairwise_distances)])
    min_distance = np.min(pairwise_distances[np.isfinite(pairwise_distances)])
    avg_min_distance = np.mean(np.min(pairwise_distances, axis=-1))
    return avg_distance, min_distance, avg_min_distance


def compute_tf_idf_similarity(texts1, texts2):
    combined_texts = texts1 + texts2

    # Vectorize the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Split the TF-IDF matrix back into two parts
    tfidf_matrix1 = tfidf_matrix[:len(texts1)]
    tfidf_matrix2 = tfidf_matrix[len(texts1):]

    # Compute cosine similarity between the two parts
    similarity_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    # From Zoe
    # wasserstein_dist_centroids = np.mean(np.array([wasserstein_distance(u, v) for u, v in
    #                                                zip(selected_topics_centroids_weighted_ngram,
    #                                                    nmf_topic_probs_centroids)]))

    # Average the pairwise similarities
    average_similarity = np.mean(similarity_matrix)

    return average_similarity


def compute_pairwise_tf_idf_similarity(texts1, texts2):
    assert len(texts1) == len(texts2)
    combined_texts = texts1 + texts2

    # Vectorize the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Split the TF-IDF matrix back into two parts
    tfidf_matrix1 = tfidf_matrix[:len(texts1)]
    tfidf_matrix2 = tfidf_matrix[len(texts1):]

    # Compute cosine similarity for each pair of corresponding texts
    similarity_scores = []
    for vec1, vec2 in zip(tfidf_matrix1, tfidf_matrix2):
        similarity = cosine_similarity(vec1, vec2)[0][0]
        similarity_scores.append(similarity)
    # From Zoe
    # wasserstein_dist_centroids = np.mean(np.array([wasserstein_distance(u, v) for u, v in
    #                                                zip(selected_topics_centroids_weighted_ngram,
    #                                                    nmf_topic_probs_centroids)]))

    # Aggregate similarity scores
    average_similarity = np.mean(similarity_scores)
    median_similarity = np.median(similarity_scores)
    summary = {
        'average_similarity': average_similarity,
        'median_similarity': median_similarity,
        'min_similarity': np.min(similarity_scores),
        'max_similarity': np.max(similarity_scores),
        'std_dev_similarity': np.std(similarity_scores),
        'similarity_scores': similarity_scores
    }
    return summary


def compute_inner_similarity(texts, n_splits=50):
    texts = copy.deepcopy(texts)
    similarities = []

    for _ in range(n_splits):
        # Randomly shuffle and split the texts into two halves
        random.shuffle(texts)
        half_size = len(texts) // 2
        texts1 = texts[:half_size]
        texts2 = texts[half_size:]

        # Average the pairwise similarities
        average_similarity = compute_tf_idf_similarity(texts1, texts2)
        similarities.append(average_similarity)

    # Calculate the average similarity over all splits
    overall_average_similarity = np.mean(similarities)
    return overall_average_similarity


def compute_pairwise_inner_similarity(texts, shuffle_words=False):
    texts = copy.deepcopy(texts)
    similarities = []

    texts1 = []
    texts2 = []
    for text in texts:
        # Split the string into words
        words = text.split()

        if shuffle_words:
            random.shuffle(words)

        # Calculate the midpoint
        midpoint = len(words) // 2

        # Split the words list into two halves
        first_half = ' '.join(words[:midpoint])
        second_half = ' '.join(words[midpoint:])

        texts1.append(first_half)
        texts2.append(second_half)

    return compute_pairwise_tf_idf_similarity(texts1, texts2)


def duplicate_real_data(real_data, real_pid, synthetic_pid):
    results = []
    for pid in synthetic_pid:
        real_idx = real_pid.index(pid)
        results.append(real_data[real_idx])
    return results


if __name__ == '__main__':
    model_name = 'bert' # 'sentence_bert'
    split = 'train'
    synthetic_file = f'/data/synthetic_DAIC/synthetic_{split}.json'
    real_file = f'/data/DAIC/{split}.json'

    synthetic_data, synthetic_scores, synthetic_original_ids = read_data_from_json_file(synthetic_file, text_tag=['Synopsis'])
    real_data, real_scores, real_ids = read_data_from_json_file(real_file, text_tag=['Synopsis'])

    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)

    synthetic_save_path = os.path.join(save_path, f'{model_name}_synthetic_embedding.npy')
    real_save_path = os.path.join(save_path, f'{model_name}_real_embedding.npy')

    if os.path.exists(synthetic_save_path) and os.path.exists(real_save_path):
        synthetic_embedding = np.load(synthetic_save_path)
        real_embedding = np.load(real_save_path)
    else:
        if model_name == 'bert':
            # Load pre-trained BERT model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
        else:
            tokenizer = None
            # Load pre-trained Sentence-BERT model
            model = SentenceTransformer('bert-base-nli-mean-tokens')

        synthetic_embedding = embed_data(model, tokenizer, synthetic_data)
        real_embedding = embed_data(model, tokenizer, real_data)

        np.save(synthetic_save_path, synthetic_embedding)
        np.save(real_save_path, real_embedding)

    print(f'Synthetic data shape is {synthetic_embedding.shape}')
    print(f'Real data shape is {real_embedding.shape}')

    all_embeddings = np.vstack((synthetic_embedding, real_embedding))
    labels = np.array([0] * len(synthetic_embedding) + [1] * len(real_embedding))
    scores = synthetic_scores + real_scores

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], label='Synthetic', alpha=0.6, c='b')
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], label='Original', alpha=0.6, c='r')
    plt.legend()
    plt.title("t-SNE visualization of paragraph embeddings")
    plt.savefig(os.path.join(save_path, 'tsne.png'))
    # plt.show()

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], label='Original', alpha=0.8, c='#a3acf8')
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], label='Synthetic', alpha=0.8, c='#ea9f97')

    # Add the scores as annotations
    # for i, score in enumerate(scores):
    #     plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(score), fontsize=9, ha='right')

    plt.legend()
    plt.title("PCA visualization of paragraph embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(os.path.join(save_path, 'pca.png'))
    # plt.show()

    ####################### Quantitative ##############################
    # TF-IDF similarity
    synthetic_similarity = compute_inner_similarity(synthetic_data)
    real_similarity = compute_inner_similarity(real_data)
    pairwise_real_similarity = compute_pairwise_inner_similarity(real_data, shuffle_words=True)
    duplicate_real = duplicate_real_data(real_data, real_ids, synthetic_original_ids)
    within_similarity = compute_pairwise_tf_idf_similarity(synthetic_data, duplicate_real)
    print(f"TF-IDF similarity within real data: {real_similarity:.04f}")
    print(f"Pairwise TF-IDF similarity within real data: {pairwise_real_similarity['average_similarity']:.04f}; Max similarity: {pairwise_real_similarity['max_similarity']:.04f}")
    # print(f"TF-IDF similarity within synthetic data: {synthetic_similarity:.04f}")
    print(f"TF-IDF similarity between real and synthetic data: {within_similarity['average_similarity']:.04f}; Max similarity: {within_similarity['max_similarity']:.04f}")
    print('#####################################################################')

    max_idx = np.argmax(within_similarity['similarity_scores'])
    print("Closest texts:")
    print("Original:")
    print(duplicate_real[max_idx])
    print('Synthetic:')
    print(synthetic_data[max_idx])
    print('#####################################################################')

    ######################## Distances ################################
    # Calculate distances within synthetic group
    # avg_euclidean_within_synthetic, min_euclidean_within_synthetic = calculate_distances(synthetic_embedding, 'euclidean')
    # avg_cosine_within_synthetic, min_cosine_within_synthetic = calculate_distances(synthetic_embedding, 'cosine')

    # Calculate distances within real group
    avg_euclidean_within_real, min_euclidean_within_real, avg_min_euclidean_within_real = calculate_distances(real_embedding, 'euclidean')
    avg_cosine_within_real, min_cosine_within_real, avg_min_cosine_within_real = calculate_distances(real_embedding, 'cosine')

    # Calculate distances between synthetic and real groups
    euclidean_between_groups = cdist(synthetic_embedding, real_embedding, 'euclidean')
    cosine_between_groups = cdist(synthetic_embedding, real_embedding, 'cosine')

    avg_euclidean_between = np.mean(euclidean_between_groups)
    min_euclidean_between = np.min(euclidean_between_groups)
    avg_min_euclidean_between = np.mean(np.min(euclidean_between_groups, axis=-1))
    avg_cosine_between = np.mean(cosine_between_groups)
    min_cosine_between = np.min(cosine_between_groups)

    print(f"Average Euclidean Distance within Real: {avg_euclidean_within_real}")
    print(f"Average Euclidean Distance between Groups: {avg_euclidean_between}")

    print('---------------------------------------------------------------------')

    print(f"Minimum Euclidean Distance within Real: {min_euclidean_within_real}")
    print(f"Minimum Euclidean Distance between Groups: {min_euclidean_between}")
    print(f"Average Minimum Euclidean Distance within Real: {avg_min_euclidean_within_real}")
    print(f"Average Minimum Euclidean Distance between Groups: {avg_min_euclidean_between}")

    print('#####################################################################')

    # print(f"Average Cosine Distance within Real: {avg_cosine_within_real}")
    # print(f"Average Cosine Distance within Synthetic: {avg_cosine_within_synthetic}")
    # print(f"Average Cosine Distance between Groups: {avg_cosine_between}")
    #
    # print('---------------------------------------------------------------------')
    #
    # print(f"Minimum Cosine Distance within Real: {min_cosine_within_real}")
    # print(f"Minimum Cosine Distance within Synthetic: {min_cosine_within_synthetic}")
    # print(f"Minimum Cosine Distance between Groups: {min_cosine_between}")
    #
    # print('#####################################################################')

    # Find the index of the minimum value in the distance matrix
    # min_index = np.unravel_index(np.argmin(euclidean_between_groups, axis=None), euclidean_between_groups.shape)
    #
    # # Extract the row and column indices
    # min_row, min_col = min_index
    #
    # print(f"Closest texts in min Euclidean distance: ({min_row}, {min_col})")
    # print("Original:")
    # print(real_data[min_col])
    # print('Synthetic:')
    # print(synthetic_data[min_row])

    print('#####################################################################')
