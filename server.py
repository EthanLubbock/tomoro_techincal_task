from flask import Flask, render_template, request, jsonify
import pandas as pd

from query_model import QueryAnswerModel

app = Flask(__name__)
qa_model = QueryAnswerModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_question', methods=['POST'])
def ask_question():
    # Read question and history from form
    data = request.form
    question = data.get("question")
    question_history = data.getlist("question_history[]")
    answer_history = data.getlist("answer_history[]")
    # Combine questions and answers as per finetuning prompts
    history = [(q, a) for q, a in zip(question_history, answer_history)]

    # Read context (table) from form
    context = request.files.get("context")
    context = pd.read_csv(context, index_col=0)
    context = context.to_string()

    # Format the input prompt with question, context, and history
    input_text = qa_model.format_prompt(question, context, history)

    # Generate answer
    answer = qa_model.query_model(input_text)

    # Return the answer in JSON format
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run()