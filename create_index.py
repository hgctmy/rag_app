from openai import OpenAI
import faiss
import numpy as np
import sqlite3
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer
import json
import os

# OpenAI APIキー設定
config = json.load(open("config.json"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# OpenAIの埋め込みモデルを使ってテキストをベクトルに変換する関数
def embed_text_openai(texts):
    batch_size = 16  # APIの利用制限を回避するためにバッチ処理
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [result.embedding for result in response.data]
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


# FAISSインデックスの作成
def build_faiss_index(embedding_matrix):
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)  # コサイン類似度を使用
    index.add(embedding_matrix)
    return index


# ドキュメントデータの読み込み
dbname = 'scraped_data_without_tag_20241017.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

cur.execute('SELECT id, content_text FROM scraped_info')


# チャンキング関数
def split_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "、", " ", ""],  # チャンクの区切り文字リスト
        chunk_size=500,           # チャンクの最大文字数
        chunk_overlap=100,         # チャンク間の重複する文字数
        length_function=len,      # 文字数で分割
        is_separator_regex=False,  # separatorを正規表現として扱う場合はTrue
    )
    chunks = text_splitter.create_documents([text])
    chunks = [line.page_content for line in chunks]
    return chunks


def clean_text(text):
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text).strip()
    # 特殊文字を削除または置換
    text = re.sub(r'[\t\r\u3000]', '', text)
    return text


# ドキュメントのチャンキング
documents = []
for row in cur.fetchall():
    id = row[0]
    text = clean_text(row[1])
    chunks = split_into_chunks(text)
    documents.extend(chunks)
documents = list(set(documents))

# チャンキングしたものを保存
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)


cur.close()
conn.close()


# ドキュメントをベクトルに変換してFAISSインデックスを作成
document_embeddings = embed_text_openai(documents)
index = build_faiss_index(document_embeddings)

faiss.write_index(index, "faiss_index_file.index")


# BM25による検索のために、単語をトークン化
tokenizer = Tokenizer()


def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]


tokenized_documents = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# BM25のモデルを保存
with open("bm25_model.pkl", 'wb') as f:
    pickle.dump(bm25, f)

# トークン化済み文書を保存
with open("tokenized_doc.pkl", 'wb') as f:
    pickle.dump(tokenized_documents, f)
