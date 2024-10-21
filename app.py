from flask import Flask, request
import glob
import rag
import jsonlines

app = Flask(__name__)


# 対話履歴を格納するディレクトリ
dialogue_dir = "./log/"


@app.route("/echo", methods=["POST"])
def echo():
    """
    {
        "data": "echoする文字列"
    }
    """

    # 送られてきたリクエストのdataキーから要素を取得
    text = request.json["data"]

    return text


@app.route("/generate-response", methods=["POST"])
def generate_chatgpt_response():
    """
    {
        "session_key": "",
        "model": "",
        "user_input": ""
    }
    """
    session_key = request.json["session_key"]
    model = request.json["model"]
    user_input = request.json["user_input"]

    session_path = f'{dialogue_dir}{session_key}.jsonl'

    context_file = glob.glob(session_path)
    context = []
    if len(context_file) == 0:
        with open(session_path, mode="w", encoding="utf-8") as f:
            f.write("")
        context.append(
            {
                "role": "ユーザ",
                "content": user_input
            }
        )
        context_file = [session_path]
    else:
        with jsonlines.open(context_file[0], mode='r') as reader:
            for entry in reader:
                context.append(entry)
        context.append(
            {
                "role": "ユーザ",
                "content": user_input
            }
        )

    response, _ = rag.rag_pipeline(model, context)
    context.append(
        {
            "role": "システム",
            "content": response
        }
    )
    with jsonlines.open(context_file[0], mode='w') as writer:
        writer.write_all(context)

    return {"response": response}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
