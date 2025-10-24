import pandas as pd
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values


class DB:
    def __init__(self, database, user, password, host, port):
        # Construct the database URL from connection parameters
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        # Create a SQLAlchemy Engine
        self.engine = create_engine(db_url)
        self.connection = None  # Kept for backward compatibility; normally use engine directly

    def executeQuery(self, queryType, query):
        # Use 'with' statement to automatically handle connection lifecycle
        with self.engine.connect() as conn:
            if queryType.lower() == "select":
                result = conn.execute(text(query))
                return result.fetchall()
            
            elif queryType.lower() in ["insert", "update"]:
                conn.execute(text(query))
                conn.commit()  # In SQLAlchemy 2.x, an explicit commit is required

    def connect(self):
        # Engine is already initialized; this method only tests the connection
        try:
            self.connection = self.engine.connect()
            print("Database connection successful.")
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.connection = None

    def executeMany(self, query, values):
        with self.engine.connect() as conn:
            # executemany accepts a list of tuples or dictionaries
            conn.execute(text(query), values)
            conn.commit()
        
    def executeValues(self, query, values):
        # execute_values is specific to psycopg2, so we obtain a raw connection
        with self.engine.connect() as conn:
            # Get a psycopg2 raw connection from SQLAlchemy connection
            raw_conn = conn.raw_connection()
            try:
                with raw_conn.cursor() as cursor:
                    execute_values(cursor, query, values)
                raw_conn.commit()
            finally:
                raw_conn.close()  # raw_connection must be closed manually
        
    def executeDict(self, query: str) -> pd.DataFrame:
        # ✅ Wrap the query string with text() automatically
        # This allows the caller to simply pass a plain SQL string
        return pd.read_sql_query(text(query), self.engine)

    
    def is_connected(self) -> bool:
        # Attempt to connect to verify database accessibility
        try:
            with self.engine.connect() as conn:
                return True
        except Exception:
            return False

    def close(self):
        # Dispose of the connection pool managed by the engine
        if self.engine:
            self.engine.dispose()
        if self.connection and not self.connection.closed:
            self.connection.close()

    # Maintain compatibility with existing method names
    def closeConnection(self):
        self.close()

    def disconnection(self):
        self.close()
