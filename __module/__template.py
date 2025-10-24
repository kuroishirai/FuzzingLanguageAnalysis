import requests
import os
import sys
from datetime import datetime, date, timedelta
from google.cloud import storage
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# ==============================================================================
# ユーザー定義モジュールのインポート
# ==============================================================================
from configparser import ConfigParser

module_folder = '/work/tatsuya-shi/research/__module'

print('[module_path]',module_folder)

if module_folder not in sys.path:
    sys.path.append(module_folder)
import utils
from dbFile import DB

config_path = module_folder + "/envFile.ini"
print('[config_path]',config_path)



# ==============================================================================
# データ収集関数（このセクションは変更なし）
# ==============================================================================


# script_dir は起動プログラムがある場所
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath('.')


def fetch_data(db):
    
    query = f"""
    SELECT project
    FROM issue_report
    """
    
    projects = db.executeQuery("select",query)  # リストで取得
    projects = db.executeDict(query)    # データフレームで取得
    



def process(db):
                            
    # 設定変数（テキストサイズ， 表示の有無， 出力先など）を保持した辞書型の定義
    config = {"output_dir": '/work/tatsuya-shi/research/FuzzingEffectiveness/data/language',
              "script_dir": script_dir,
              }
    
    # 関数の呼び出し
    data = fetch_data(db)

# ==============================================================================
# メイン実行部分 (utils.py を活用するように修正)
# ==============================================================================

def main(task_id: int):
    """
    SLURMのタスクIDに基づき、処理対象プロジェクトを決定し、結果をDBに保存する。
    """
    start_time = utils.return_now_datetime_jst()
    utils.save_to_file(f'task_id: {task_id}')
    
    # --- データベース接続設定 (utils.py を使用) ---
    db = utils.setup_db(config_path)
    db.connect()
    
    process(db)
    
    
    
    
    utils.notify_slack(f"{os.path.basename(__file__)}: {task_id}", start_time = start_time)

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 1:
        main(0)
    elif len(sys.argv) < 2:
        print("実行方法: python get_reachability_data.py <SLURM_ARRAY_TASK_ID>")
        print("例: python get_reachability_data.py 1")
        sys.exit(1)
    else:
        try:
            task_id = int(sys.argv[1])
        except ValueError:
            print(f"エラー: タスクIDは整数である必要があります。入力: '{sys.argv[1]}'")
            sys.exit(1)

        main(task_id)
