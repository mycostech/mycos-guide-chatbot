import psycopg
from psycopg import sql

from lib.config import load_config

config = load_config()

def load_vector_data():
    # --- 1. Load data from PostgreSQL Vector DB ---
    X = [] # To store the embeddings (features)
    y = [] # To store the names (labels)

    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg.connect(config['db_connection'])
        cur = conn.cursor()

        # Construct the SQL query safely using sql.SQL and sql.Identifier
        # This prevents SQL injection and properly handles column/table names.
        query = sql.SQL("SELECT {name_col}, {embedding_col} FROM {table_name}").format(
            name_col=sql.Identifier("name"),
            embedding_col=sql.Identifier("embedding"),
            table_name=sql.Identifier("documents")
        )

        # Execute the query
        cur.execute(query)

        # Fetch all rows
        rows = cur.fetchall()

        if not rows:
            print("No data found in the 'documents' table. Please ensure your table has data.")
            exit()

        # Process fetched rows
        for row in rows:
            label, embedding_data = row
            y.append(label)

            # Assuming 'embedding_data' comes as a list or array-like string from your DB.
            # If your 'embedding' column is a text type storing "[1.2, 3.4, ...]", you'll need to parse it.
            # If it's a native vector type or array type, psycopg2 might handle it directly as a list/tuple.
            # Example for parsing a string representation like "[1.2, 3.4, 5.6]"
            if isinstance(embedding_data, str):
                # Remove brackets and split by comma, then convert to float
                embedding_list = [float(x.strip()) for x in embedding_data.strip('[]').split(',')]
                X.append(embedding_list)
            else:
                # If it's already a list/tuple (e.g., from a native PG array/vector type)
                X.append(embedding_data)
    except psycopg.Error as e:
        print(f"Error connecting to or querying the database: {e}")
        exit()
    finally:
        # Close the cursor and connection
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

    return (X, y)