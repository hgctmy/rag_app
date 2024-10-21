from openai import OpenAI
import faiss
import numpy as np
from janome.tokenizer import Tokenizer
import pickle
import json

# OpenAI APIキー設定
config = json.load(open("config.json"))
client = OpenAI(api_key=config["openai_api_key"])

# ドキュメントとfaissインデックス，bm25モデルを準備
index = faiss.read_index("faiss_index_file.index")
documents = pickle.load(open("documents.pkl", "rb"))
bm25 = pickle.load(open("bm25_model.pkl", "rb"))
tokenized_documents = pickle.load(open("tokenized_doc.pkl", "rb"))

# 日本語トークン化のためのJanomeトークナイザのインスタンスを作成
tokenizer = Tokenizer()


# BM25による検索のために、単語をトークン化
def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]


# OpenAI Embeddingsを用いてベクトル化
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return embeddings


def rrf_score(rank, k=60):
    return 1 / (k + rank)


# RRFでbm25とfaissのスコアを組み合わせる
def rrf_rerank(faiss_results, bm25_results, k):
    scores = {}

    for rank, (doc_id, _) in enumerate(faiss_results):
        scores[doc_id] = scores.get(doc_id, 0) + rrf_score(rank + 1, k)

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + rrf_score(rank + 1, k)

    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return reranked


# クエリに対してBM25とベクトル検索を行い、RRFでリランキング
def hybrid_search_with_rrf(query):
    # クエリをトークン化してBM25で検索
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    # クエリをベクトル化してFAISSで検索
    query_embedding = get_embeddings([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), len(documents))
    # RRFでリランキング
    rrf_indices = rrf_rerank([(idx, score) for idx, score in top_n], list(zip(indices[0], distances[0])), 60)

    return rrf_indices


# 検索した文書をもとに回答を生成
def generate_answer(model, query, retrieved_docs):
    # 検索したドキュメントを結合して、質問と共にGPTに投げる
    context = "\n\n".join(retrieved_docs)
    prompt = f"以下のコンテキストを使用してユーザーの質問に答えてください。\nわからないことがあれば、それを明確に述べてください。\nまた、質問と直接関係がない場合は、「わからない」と答えてください。\nコンテキストを抜粋・引用して簡潔に、かつ段階的に考えて答えを生成してください。\nあなたの答えはコンテキストと質問に基づいていることを確認してください。\n回答に不足がないようにしてください。\n\n----------------\n\n{context}\n\n\n\n質問:{query}\n回答:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content


# 対話履歴を利用した検索のための質問文生成
def generate_query(model, context):
    prompt = f"以下の情報を使用してユーザーの質問に答えてください。\n与えた情報が質問と直接関係がない場合は、「私の持っている情報ではお答えすることができません。」と答えてください。\nあなたの答えは与えた情報と質問に基づいていることを確認してください。\n回答に不足がないようにしてください。\n\n----------------\n\n{context}\n\n\n\n質問:{query}\n回答:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content


# RAGのフローを実行
def rag_pipeline(model, context, documents=documents, k=5):
    if type(context) is not list:
        query = context
    elif len(context) <= 1:
        query = context[-1]["content"]
    else:
        query = generate_query(model, context)
    # FAISS検索
    rank = hybrid_search_with_rrf(query)
    # 検索されたドキュメントを取得
    indices = [doc_id for doc_id, _ in rank][:k]
    retrieved_docs = [documents[i] for i in indices]
    # ドキュメントを元に回答を生成
    answer = generate_answer(model, query, retrieved_docs)
    return answer, retrieved_docs


if __name__ == '__main__':
    # 質問を入力してRAGを実行
    query = input("Question: ")
    context = [{"role": "ユーザ", "content": query}]
    answer, doc = rag_pipeline("gpt-4o-mini", context)

    # 結果の表示
    print(f"Answer: {answer}")
    print(f"Documents: {doc}")
