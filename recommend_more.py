## Paper Recommender Part 3 ##
## Recommender Additional Five Papers Based on User's Frist Liked Paper ## 
## Using content-based, Cosine Similarity Algorithm ##

import pandas as pd
import numpy as np
from sentence_transformers import util, SentenceTransformer
import torch


embeddings = np.load("paper_embeddings.npy")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_more_from_liked_paper(liked_paper_text, label, top_k=5):
    # 找到该 label 对应的子集
    df = pd.read_csv("cleaned_papers_with_id.csv")
    mask = df["label"] == label
    label_subset_df = df[mask].reset_index(drop=True)
    label_subset_embeddings = embeddings[mask.values]

    if len(label_subset_df) <= 1:
        print("❌ Not enough papers in this label to recommend more.")
        return None

    # 嵌入喜欢的那篇论文（title+abstract）
    liked_embedding = embedding_model.encode(liked_paper_text)

    # 计算相似度
    cosine_scores = util.cos_sim(liked_embedding, label_subset_embeddings)[0]

    # 排除自己（相似度 = 1.0 的那篇），选出 top_k+1 再排除 index 0（假设相同）
    top_indices = cosine_scores.argsort(descending=True)[1:top_k+1]
    top_papers = label_subset_df.iloc[top_indices]

    for idx, row in enumerate(top_papers.itertuples(), 1):
        print(f"\n📄 Paper You Might Also Like NO.{idx}:")
        print("Paper Title:", row.original_title)
        print("Abstract:")
        print(row.original_abstract)

    return top_papers




## Paper Recommender Part 4 ##
## Recommend Alternativly Additional Five Papers Based on User's Frist Liked Recommended Paper ## 
## Using content-based, MMR-enhanced Cosine Similarity Algorithm to Ensure Diversity##

def mmr(doc_embedding, candidate_embeddings, top_k=5, lambda_param=0.5):
    selected_indices = []
    remaining_indices = list(range(len(candidate_embeddings)))

    for _ in range(top_k):
        mmr_scores = []
        for idx in remaining_indices:
            sim_to_query = util.cos_sim(doc_embedding, candidate_embeddings[idx])[0].item()
            sim_to_selected = max(
                [util.cos_sim(candidate_embeddings[idx], candidate_embeddings[j])[0].item() for j in selected_indices]
                or [0.0]
            )
            score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((idx, score))

        selected = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(selected)
        remaining_indices.remove(selected)

    return selected_indices


def alternative_recommend_more_from_liked_paper(liked_paper_text, label, top_k=5):
    df = pd.read_csv("cleaned_papers_with_id.csv")
    mask = df["label"] == label
    label_subset_df = df[mask].reset_index(drop=True)
    label_subset_embeddings = embeddings[mask.values]

    if len(label_subset_df) <= 1:
        print("❌ Not enough papers in this label to recommend more.")
        return None

    liked_embedding = embedding_model.encode(liked_paper_text)

    same_text_mask = label_subset_df["paper"] == liked_paper_text
    exclude_indices = set(label_subset_df[same_text_mask].index)

    candidate_embeddings = [embedding for i, embedding in enumerate(label_subset_embeddings) if i not in exclude_indices]
    candidate_df = label_subset_df.drop(index=exclude_indices).reset_index(drop=True)

    mmr_indices = mmr(
        doc_embedding=liked_embedding,
        candidate_embeddings=candidate_embeddings,
        top_k=top_k,
        lambda_param=0.45  # 可调成 0.3 更 diverse，0.7 更相关
    )

    alternative_top_papers = candidate_df.iloc[mmr_indices]

    for idx, row in enumerate(alternative_top_papers.itertuples(), 1):
        print(f"\n📄 Paper You Might Also Like NO.{idx}:")
        print("Paper Title:", row.original_title)
        print("Abstract:")
        print(row.original_abstract)

    return alternative_top_papers

