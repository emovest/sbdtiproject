## Paper Recommender Part 2 ##
## Recommender One Paper from the Predicted Label to the USER ##
## And Ask the User If He or She Like It or Not ##


import pandas as pd
import numpy as np
import torch
from scipy.sparse import hstack
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# === 加载三分类模型相关文件 ===
bow_vectorizer = joblib.load("champion_bow_vectorizer.pkl")
xgb_bow = joblib.load("champion_xgboost_model.pkl")
le_bow = joblib.load("champion_label_encoder.pkl")

lda_vectorizer = joblib.load("secondmodel_bow_vectorizer_LDA.pkl")
lda_model = joblib.load("secondmodel_lda_model.pkl")
xgb_bow_lda = joblib.load("secondmodel_xgboost_LDA.pkl")
le_lda = joblib.load("secondmodel_label_encoder.pkl")

bert_model = BertForSequenceClassification.from_pretrained("xrayenglish/bert_classifier_model")
bert_tokenizer = BertTokenizer.from_pretrained("xrayenglish/bert_classifier_tokenizer")
bert_trainer = Trainer(model=bert_model)
le_bert = joblib.load("bert_label_encoder.pkl")

# === 加载推荐系统数据 ===
df = pd.read_csv("cleaned_papers_with_id.csv")
embeddings = np.load("paper_embeddings.npy")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === 三模型融合预测函数 ===
def predict(text):
    # BOW + XGBoost
    X_bow = bow_vectorizer.transform([text])
    pred1 = xgb_bow.predict(X_bow)[0]
    label1 = le_bow.inverse_transform([pred1])[0]
    prob1 = xgb_bow.predict_proba(X_bow)[0]

    # BOW + LDA + XGBoost
    X_bow_lda = lda_vectorizer.transform([text])
    X_lda = lda_model.transform(X_bow_lda)
    X_combined = hstack([X_bow_lda, X_lda])
    pred2 = xgb_bow_lda.predict(X_combined)[0]
    label2 = le_lda.inverse_transform([pred2])[0]
    prob2 = xgb_bow_lda.predict_proba(X_combined)[0]

    # BERT
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    bert_model.to("cpu")
    with torch.no_grad():
        output = bert_model(**inputs)
        logits = output.logits
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred3 = np.argmax(probs)
        label3 = le_bert.inverse_transform([pred3])[0]

    # 加权平均投票（权重：BERT=1，其他=0.5）
    prob1_weighted = prob1 * 0.5
    prob2_weighted = prob2 * 0.5
    prob3_weighted = probs * 1.0

    final_prob = prob1_weighted + prob2_weighted + prob3_weighted
    final_pred = np.argmax(final_prob)
    final_label = le_bert.inverse_transform([final_pred])[0]

    return final_label


# First Paper Recommender Function
# Using content-based, Cosine Similarity Algorithm
def recommend_paper(text):
    # Step 1: 分类预测
    predicted_label = predict(text)
    print(f"Predicted Label: {predicted_label}")

    # Step 2: 找出该类论文
    mask = df["label"] == predicted_label
    label_subset_df = df[mask].reset_index(drop=True)
    label_subset_embeddings = embeddings[mask.values]

    if len(label_subset_df) == 0:
        print("❌ No papers found in this label.")
        return None

    # Step 3: 计算语义相似度
    user_embedding = embedding_model.encode(text)
    cosine_scores = util.cos_sim(user_embedding, label_subset_embeddings)[0]
    top_idx = cosine_scores.argmax()
    best_paper = label_subset_df.iloc[[top_idx]]

    # Step 4: 输出
    print("\n📄 Recommended Paper:")
    print("Paper Title: ", best_paper["original_title"].values[0])
    print("Paper Abstract: ")
    print(best_paper["original_abstract"].values[0])
    return best_paper
