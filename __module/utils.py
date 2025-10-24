import os
from configparser import ConfigParser
from dbFile import DB
import csv
import requests
import slackweb
from datetime import datetime, timezone, timedelta
from typing import List, Dict


def resolve_relative_path_from_script(relative_path: str) -> str:
    """
    Convert a relative path to an absolute path based on the script's file location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.abspath(os.path.join(script_dir, relative_path))
    return resolved_path


def save_to_file(text, filename="/work/tatsuya-shi/research/__batch/output/test.txt"):
    """
    Append the specified text to a file.  
    If the directory does not exist, it will be created automatically.
    """
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f'{text}\n')
        

def setup_db(path):
    """
    Initialize and return a database connection using settings from the given configuration file.
    """
    configObj = ConfigParser()
    configObj.read(path)
    postgresInfo = configObj["POSTGRES"]
        
    db = DB(
        database=postgresInfo["POSTGRES_DB"],
        user=postgresInfo["POSTGRES_USER"],
        password=postgresInfo["POSTGRES_PASSWORD"],
        host=postgresInfo["POSTGRES_IP"],
        port=postgresInfo["POSTGRES_PORT"]
    )
    return db  # Return the DB instance


# ==============================================================================
# ### Modified section starts here ###
# ==============================================================================
def insert_data_list_to_db(db, table_name, data_list):
    """
    Insert a list of dictionaries into the specified database table.
    Dynamically generates and executes an INSERT statement using the DB instance.

    Args:
        db: Instance of the DB class (expected to have a cursor and connection).
        table_name (str): Name of the target table.
        data_list (List[dict]): List of data entries to insert.
    """
    if not data_list:
        return

    for data in data_list:
        # Skip empty or invalid data
        if not data or not isinstance(data, dict):
            continue

        columns = data.keys()
        values = list(data.values())
        
        # Wrap column names in double quotes to avoid SQL keyword conflicts
        column_names = ', '.join(f'"{c}"' for c in columns)
        placeholders = ', '.join(['%s'] * len(values))
        
        insert_query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        try:
            # Execute the query using the cursor
            db.cursor.execute(insert_query, values)
        except Exception as e:
            # Print detailed error information
            print(f"--- DB Insert Error ---")
            print(f"Record: {data}")
            print(f"Exception: {e}")
            print("-----------------------")
            # Raise the exception so the caller can handle the transaction
            raise


def save_dict_list_to_csv(data: List[Dict], output_path: str):
    """
    Save a list of dictionaries to a CSV file.
    All keys from the list are collected as column names (union of all keys).
    Missing keys are filled with None.
    """
    # Collect all unique keys to use as column headers
    all_keys = set()
    for d in data:
        all_keys.update(d.keys())
    fieldnames = sorted(all_keys)  # Optional: sort for consistent column order

    # Write to CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            complete_row = {key: row.get(key, None) for key in fieldnames}
            writer.writerow(complete_row)
            


def return_now_datetime_jst():
    """
    Return the current datetime in JST (UTC+9).
    """
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    return now_jst


def notify_slack(message="Finish Program!", start_time=None, url='XXXXXXXXXXXXXXXXXXXXXXXXXXX'):
    """
    Send a notification message to Slack, including execution time and timestamp.
    """
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
    
    
def print_ruled_line(title, size=30, deco='='):
    """
    Print a title surrounded by decorative characters for readability.
    Example:
    ====================== Title ======================
    """
    print(deco * size, title, deco * size)
