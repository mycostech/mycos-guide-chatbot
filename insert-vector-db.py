import code
import os
from sentence_transformers import SentenceTransformer
import psycopg
from numpy import linalg as LA


files = [f for f in os.listdir(f"temp/scrape_storage") if f.endswith('.txt')]

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

conn = psycopg.connect("")


for file in files:
    cur = conn.cursor()

    # print(file)

    with open(f"temp/scrape_storage/{file}", "r", encoding="utf-8") as f:
        text = f.read()

    embedding = model.encode(text)

    # normalize embedding for better cosine distance
    norm_embedding = embedding / LA.norm(embedding)

    # code.interact(local=locals())

    # embedding type is `numpy.ndarray`, 
    cur.execute(
        "INSERT INTO documents (name, raw_text, embedding) VALUES (%s, %s, %s)",
        (file, text, norm_embedding.tolist())
    )

    conn.commit()
    print(f"inserted vector data of {file}")

cur.close()
conn.close()