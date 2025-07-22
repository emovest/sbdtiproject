from flask import Flask, request, jsonify
from upstash_redis import Redis
import os
from recommend_more import recommend_more_from_liked_paper, mmr, alternative_recommend_more_from_liked_paper
from recommender import predict, recommend_paper
import json
import pandas as pd
from bartsummary import summarize_papers_with_bart



app = Flask(__name__)

# 初始化 Upstash Redis（确保 Render 设置了这两个环境变量）
redis = Redis(
    url=os.environ.get("UPSTASH_REDIS_REST_URL"),
    token=os.environ.get("UPSTASH_REDIS_REST_TOKEN")
)

@app.route('/')
def home():
    return "✅ Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    # Dialogflow 的结构中，intent name 是在 queryResult 中
    intent = data["queryResult"]["intent"]["displayName"]
    user_input = data["queryResult"]["queryText"]
    user_id = data["session"]  # 可以简化处理

    print(f"🎯 Received intent: {intent}")

    # 如果是主推荐意图
    if intent == "getUserCrytoInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })

    # 如果是主推荐意图2
    if intent == "getUserRealEstateInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })

    # 如果是主推荐意图3
    if intent == "getUserArtsInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })


    # 如果是主推荐意图4
    if intent == "getUserGoldInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })
        
    # 如果是主推荐意图5
    if intent == "getUserQuantInterest":
        final_label = predict(user_input)
        best_paper = recommend_paper(user_input)

        # 把推荐中的第一篇的文本和标签存入 Redis
        liked_text = best_paper["paper"].values[0]
        liked_label = final_label
        redis.set(f"{user_id}:liked_text", liked_text)
        redis.set(f"{user_id}:liked_label", liked_label)
        liked_abstract = best_paper["original_abstract"].values[0]
        redis.set(f"{user_id}:liked_abstract", liked_abstract)

        return jsonify({
            "fulfillmentText": (
                f"📌 Recommended Paper: \n\n"
                f"📄 {best_paper['original_title'].values[0]}\n\n"
                f"📝 Abstract:\n\n"
                f"{best_paper['original_abstract'].values[0]}\n\n"
            )
        })
        

    # 如果是请求更多推荐的意图
    if intent == "getUserIntentforMorePaper":
        liked_text = redis.get(f"{user_id}:liked_text")
        liked_label = redis.get(f"{user_id}:liked_label")

        if liked_text is None or liked_label is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I couldn't find your previous preferences. Please tell me your research interest again."]}}
                ]
            })

        more_papers = recommend_more_from_liked_paper(liked_text, liked_label, top_k=5)
        
        # 提取摘要列表
        abstract_list = more_papers["original_abstract"].tolist()
        
        # 存入 Redis（建议用 json）
        redis.set(f"{user_id}:more_abstracts", json.dumps(abstract_list))

        response_text = "📚 Here are some more papers you might like:\n\n"
        for idx, row in enumerate(more_papers.itertuples(), 1):
            response_text += (
                f"🔹 Paper {idx}:\n"
                f"📄 Title: {row.original_title}\n"
                f"📝 Abstract: {row.original_abstract}\n"
                f"— — — — —\n\n"
            )
            
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ]
        })


    # if not satisfied
    if intent == "getUserIntentforAlternativePaper":
        liked_text = redis.get(f"{user_id}:liked_text")
        liked_label = redis.get(f"{user_id}:liked_label")

        if liked_text is None or liked_label is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I couldn't find your previous preferences. Please tell me your research interest again."]}}
                ]
            })

        mmr_cosine_recommended = alternative_recommend_more_from_liked_paper(liked_text, liked_label, top_k=5)
        
        # 提取摘要列表
        abstract_list = mmr_cosine_recommended["original_abstract"].tolist()
        
        # 存入 Redis（建议用 json）
        redis.set(f"{user_id}:more_abstracts", json.dumps(abstract_list))
        
        response_text = "📚 Here are some alternative papers you might like:\n\n"
        for idx, row in enumerate(mmr_cosine_recommended.itertuples(), 1):
            response_text += (
                f"🔹 Paper {idx}:\n"
                f"📄 Title: {row.original_title}\n"
                f"📝 Abstract: {row.original_abstract}\n"
                f"— — — — —\n\n"
            )
            
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ]
        })



    # Get summary
    elif intent == "getSummary":
        liked_abstract = redis.get(f"{user_id}:liked_abstract")
        more_abstracts = redis.get(f"{user_id}:more_abstracts")
        
        if liked_abstract is None or more_abstracts is None:
            return jsonify({
                "fulfillmentMessages": [
                    {"text": {"text": ["⚠️ Sorry, I need both the liked and recommended abstracts to generate a summary."]}}
                ]
            })
        
        # 解析回列表
        all_abstracts = [liked_abstract] + json.loads(more_abstracts)

        # 转成 DataFrame 结构
        df_to_summarize = pd.DataFrame(all_abstracts, columns=["original_abstract"])

        # 调用 summarization 函数
        summary_text = summarize_papers_with_bart(df_to_summarize)

        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": [f"📚 Summary of Selected Papers:\n\n{summary_text}"]}}
            ]
        })

    

    
    # 兜底情况
    else:
        return jsonify({
            "fulfillmentMessages": [
                {"text": {"text": ["❓ I didn't quite understand that."]}}
            ]
        })

if __name__ == '__main__':
    app.run(debug=True)
