import datetime
import json
from sentence_transformers import SentenceTransformer
from numpy import linalg as LA
from ollama import chat
from ollama import ChatResponse

import psycopg

from lib.config import load_config

TOP_MATCH_COUNT = 3

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
config = load_config()

def chat_loop():
    question = input("Question: ")
    question_embedding = model.encode(question) 
    norm_question_embedding = (question_embedding / LA.norm(question_embedding)).tolist()

    vector_literal = "[" + ",".join([str(x) for x in norm_question_embedding]) + "]"

    conn = psycopg.connect(config['db_connection'])

    cur = conn.cursor()
    sql = """SELECT
        id,
        name,
        raw_text,
        embedding <=> %s
    FROM
        documents
    ORDER BY
        embedding <=> %s
    LIMIT 10;
    """

    results = cur.execute(sql, (vector_literal, vector_literal))
    context = []

    for i in range(TOP_MATCH_COUNT):
        row = results.fetchone()
        context.append(row[2])

    prompt = f"""You're Mycos Technologies Company Promoter, please answer this question.

    Question: {question}

    Here is provided context:

    {"\n\n".join(context)}
    """

    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])

    chat_response = response.message.content

    print(chat_response)

    conn.close()

    log = {
        "question": question,
        "context": context,
        "chat_response": chat_response,
    }
    log_filename = f"log/chatlog_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json"
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(log, indent=2))

if __name__ == "__main__":
    while True:
        chat_loop()