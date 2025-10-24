import pandas as pd
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values


class DB:
    def __init__(self, database, user, password, host, port):
        # 接続情報からデータベースURLを構築
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        # SQLAlchemy の Engine を作成
        self.engine = create_engine(db_url)
        self.connection = None # 互換性のために残すが、基本は engine を使う

    def executeQuery(self, queryType, query):
        # Engineから接続を取得し、自動で解放する 'with' 構文を使用
        with self.engine.connect() as conn:
            if queryType.lower() == "select":
                result = conn.execute(text(query))
                return result.fetchall()
            
            elif queryType.lower() in ["insert", "update"]:
                conn.execute(text(query))
                conn.commit() # SQLAlchemy 2.x では commit が必要

    def connect(self):
        # Engineは初期化時に設定済み。接続テストを行う。
        try:
            self.connection = self.engine.connect()
            print("Database connection successful.")
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.connection = None

    def executeMany(self, query, values):
        with self.engine.connect() as conn:
            # executemany はタプルや辞書のリストを受け取る
            conn.execute(text(query), values)
            conn.commit()
        
    def executeValues(self, query, values):
        # execute_values は psycopg2 固有機能のため、raw接続を取得して使う
        with self.engine.connect() as conn:
            # SQLAlchemy Connectionからpsycopg2のraw connectionを取得
            raw_conn = conn.raw_connection()
            try:
                with raw_conn.cursor() as cursor:
                    execute_values(cursor, query, values)
                raw_conn.commit()
            finally:
                raw_conn.close() # raw_connection は手動で閉じる
        
    def executeDict(self, query: str) -> pd.DataFrame:
        # ✅ query を text() でラップしてから渡すように変更
        # これにより、呼び出し元は普通の文字列を渡すだけで良くなる
        return pd.read_sql_query(text(query), self.engine)

    
    def is_connected(self) -> bool:
        # 接続を試みることで、接続可能かテストする
        try:
            with self.engine.connect() as conn:
                return True
        except Exception:
            return False

    def close(self):
        # Engineが管理する接続プールを破棄する
        if self.engine:
            self.engine.dispose()
        if self.connection and not self.connection.closed:
            self.connection.close()

    # 既存のメソッド名との互換性を保つ
    def closeConnection(self):
        self.close()

    def disconnection(self):
        self.close()