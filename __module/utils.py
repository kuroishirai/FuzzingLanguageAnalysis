import os
from configparser import ConfigParser
from dbFile import DB
import csv
import requests
import slackweb
from datetime import datetime, timezone, timedelta
import csv
from typing import List, Dict

import slackweb
from datetime import datetime, timedelta
from datetime import timezone


def resolve_relative_path_from_script(relative_path: str) -> str:
    """
    Pythonスクリプトのファイル位置を基準に、相対パスを絶対パスに変換する。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.abspath(os.path.join(script_dir, relative_path))
    return resolved_path


def save_to_file(text, filename="/work/tatsuya-shi/research/__batch/output/test.txt"):
    """
    指定された文字列をテキストファイルに追記保存する関数
    """
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f'{text}\n')
        

def setup_db(path):
    configObj = ConfigParser()
    configObj.read(path)
    postgresInfo = configObj["POSTGRES"]
        
    db = DB(database=postgresInfo["POSTGRES_DB"], user=postgresInfo["POSTGRES_USER"],
            password=postgresInfo["POSTGRES_PASSWORD"], host=postgresInfo["POSTGRES_IP"],
            port=postgresInfo["POSTGRES_PORT"])
    return db # DBインスタンスを返すように修正（もし未修正の場合）

# ==============================================================================
# ### 変更点: ここから ###
# ==============================================================================
def insert_data_list_to_db(db, table_name, data_list):
    """
    辞書型リストのデータを指定されたDBテーブルに挿入する。
    DBクラスが持つカーソルを直接利用してINSERT文を動的に生成・実行する。

    Args:
        db: DBクラスのインスタンス (cursorとconnectionを持つことを想定)
        table_name (str): 挿入先のテーブル名
        data_list (list[dict]): データのリスト。
    """
    if not data_list:
        return

    for data in data_list:
        # 空の辞書や不正なデータはスキップ
        if not data or not isinstance(data, dict):
            continue

        columns = data.keys()
        values = list(data.values())
        
        # カラム名をダブルクォートで囲み、SQL予約語との衝突を回避
        column_names = ', '.join(f'"{c}"' for c in columns)
        placeholders = ', '.join(['%s'] * len(values))
        
        insert_query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        try:
            # カーソルを使ってクエリを実行
            db.cursor.execute(insert_query, values)
        except Exception as e:
            # エラー発生時に詳細な情報を出力
            print(f"--- DB Insert Error ---")
            print(f"Record: {data}")
            print(f"Exception: {e}")
            print("-----------------------")
            # トランザクションは呼び出し元で管理するため、例外を再送出
            raise


def save_dict_list_to_csv(data: List[Dict], output_path: str):
    # 全てのキーを集めて列名とする（すべての辞書のキーの和集合）
    all_keys = set()
    for d in data:
        all_keys.update(d.keys())
    fieldnames = sorted(all_keys)  # 任意でソート

    # 書き込み
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # 欠損しているキーは None として補完
            complete_row = {key: row.get(key, None) for key in fieldnames}
            writer.writerow(complete_row)
            
            


def return_now_datetime_jst():
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    return now_jst

def notify_slack(message="Finish Program!", start_time=None,url='XXXXXXXXXXXXXXXXXXXXXXXXXXX'):
    slack = slackweb.Slack(url=url)
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    print('='*20, 'Finish', '='*20)
    if start_time:
        elapsed_time = datetime.now(JST) - start_time
        
    else:
        elapsed_time = None
    slack.notify(text=f"XXXXXX {message}\n{elapsed_time}\n({now_jst.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{message}\n{elapsed_time}\n({now_jst.strftime('%Y-%m-%d %H:%M:%S')})")
    
    
def print_ruled_line(title, size = 30, deco = '='):
    print(deco*size, title, deco*size)
