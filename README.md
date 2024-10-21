# RAG (Retrieval-Augmented Generation) Flask アプリ

このアプリは、Flaskを使用して構築された簡単なRAG（Retrieval-Augmented Generation）アプリです。アプリは、事前に用意されたコーパス（知識ベース）から関連するドキュメントを取得し、生成モデルを使ってユーザーの質問に回答します。

## 特徴
- **ドキュメント検索**: FAISSやBM25などの検索モデルを使用して、コーパスから関連するドキュメントを取得。
- **テキスト生成**: 取得したドキュメントに基づいて、生成モデル（例：OpenAIのGPT）を使用して回答を生成。


### セットアップ手順

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
   

5. **データを準備**:
   - コーパスを`data/`フォルダに配置します（または任意のディレクトリを指定してください）。
   - FAISSやBM25などの検索モデルが、ドキュメントをインデックス化し、検索できるように設定します。
   - データの読み込み方法に関しては適宜変更してください。

6. **アプリを実行**:
   ```bash
   python app.py
   ```

7. **質問を行う**:
    "行いたい質問"の部分を書き換え、以下のコマンドを実行してください。
   ```bash
   curl -X POST http://127.0.0.1:8000/generate-response -H "Content-Type: application/json" -d '{"session_key": "test", "model": "gpt-4o-mini", "user_input": "行いたい質問"}'
   ```
    または、`test_post.py`の"行いたい質問"の部分を書き換え、実行してください。
   ```python
   python test_post.py
   ```


