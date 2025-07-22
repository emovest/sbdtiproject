## Summarization ##
## Providing Summarization of the abstracts of User's Liken Papers ##
## Using DistilBART Model ##

from transformers import pipeline

# ✅ 一次性加载模型，无懒加载
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_papers_with_bart(papers_df, text_column="original_abstract", max_tokens=400, min_tokens=200):
    abstracts = papers_df[text_column].tolist()
    full_text = " ".join(abstracts)

    # ✅ 安全截断到 1024 tokens 内（DistilBART 的硬上限）
    if len(full_text.split()) > 1024:
        full_text = " ".join(full_text.split()[:1024])

    # ✅ 生成摘要
    summary = summarizer(
        full_text,
        max_length=max_tokens,
        min_length=min_tokens,
        do_sample=False
    )[0]["summary_text"]

    return summary
