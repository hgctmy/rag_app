# RAG (Retrieval-Augmented Generation) Flask アプリ

このアプリは、Flaskを使用して構築された簡単なRAG（Retrieval-Augmented Generation）アプリです。アプリは、web API になっており、事前に用意されたドキュメントから関連するドキュメントを取得し、生成モデルを使ってユーザーの質問に回答します。

## 特徴
- **ドキュメント検索**: FAISSやBM25などの検索モデルを使用して、コーパスから関連するドキュメントを取得。
- **テキスト生成**: 取得したドキュメントに基づいて、生成モデル（例：OpenAIのGPT）を使用して回答を生成。


### 利用手順

1. **リポジトリをクローン**:
   ```bash
   git clone https://github.com/yourusername/rag-flask-app.git
   ```


2. **依存関係をインストール**:
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI APIキーを設定** (OpenAIを使用する場合):
   ディレクトリ内に`config.json`ファイルを作成し、以下のようにAPIキーを記入してください。:
   ```config.json
   { "openai_api_key": your_openai_api_key}
   ```
   

4. **データを準備**:
   - コーパスを`data/`フォルダに配置します（または任意のディレクトリを指定してください）。
   - データの読み込み方法に関しては適宜`create_index.py`を変更してください。

5. **検索用インデックスの作成**:
   以下のコマンドを実行してください。
   ```bash
   python create_index.py
   ```
  
6. **検索方法の設定**:
   - デフォルトではOpenAIembeddingsとFAISSを使った密なベクトル検索を用いる設定になっています。
   - キーワードベースの検索手法bm25と組み合わせた検索を行いたい場合は，`app.py`内の`rag.py`に関する部分を`rag_hybrid.py`をインポートして使うように変更してください。


7. **アプリを実行**:
   ```bash
   python app.py
   ```

8. **質問を行う**:
    "行いたい質問"の部分を書き換え、以下のコマンドを実行してください。
   ```bash
   curl -X POST http://127.0.0.1:8000/generate-response -H "Content-Type: application/json" -d '{"session_key": "test", "model": "gpt-4o-mini", "user_input": "行いたい質問"}'
   ```
    または、`test_post.py`の"行いたい質問"の部分を書き換え、実行してください。
   ```python
   python test_post.py
   ```


### 補足

利用手順の6. まで実行したのち、`rag.py`または`rag_hybrid.py`を実行することでも利用できます。
