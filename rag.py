from openai import OpenAI
import faiss
import numpy as np
import pickle
import json
import datetime

# OpenAI APIキーの設定
config = json.load(open("config.json"))
client = OpenAI(api_key=config["openai_api_key"])


# 文章をベクトル化
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return np.array(embeddings)


# FAISS検索機能
def search_faiss_index(query, index, k=5):
    query_vector = get_embeddings([query])
    distances, indices = index.search(query_vector, k)
    return distances, indices


# FAISSのインデックスとドキュメントを読み込む
index = faiss.read_index("faiss_index_file.index")
documents = pickle.load(open("documents.pkl", "rb"))


# OpenAIのGPTモデルを使って回答を生成する関数
def generate_answer(model, query, retrieved_docs):
    # 検索したドキュメントを結合して、質問と共にGPTに投げる
    context = "\n\n".join(retrieved_docs)
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


def generate_query(model, context):
    prompt = f"以下の対話履歴を参照して、ユーザが現在聞きたい内容を質問文としてまとめてください。\n\n----------------\n\n{context}\n\n\n\n質問:"
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


# クエリに基づいてRAGのフローを実行
def rag_pipeline(model, context, documents=documents, k=5):
    if type(context) is not list:
        query = context
    elif len(context) <= 1:
        query = context[-1]["content"]
    else:
        query = generate_query(model, context)
    today = datetime.date.today()
    tommorow = today + datetime.timedelta(days=1)
    query = query.replace("今日", str(today))
    query = query.replace("明日", str(tommorow))
    # FAISS検索
    distances, indices = search_faiss_index(query, index, k)
    # 検索されたドキュメントを取得
    retrieved_docs = [documents[i] for i in indices[0]]
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
    print(doc)
