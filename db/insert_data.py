import sqlite3  

conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

def convert_list_to_tuple(job_ids):
    return [(job_id,) for job_id in job_ids]

def insert_all_fetched_job_ids(job_ids):
    
    sql_for_creating_table = """
    CREATE TABLE IF NOT EXISTS job_processing_status (
       job_id TEXT UNIQUE,
       is_scraped BOOLEAN DEFAULT 0 NOT NULL CHECK (is_scraped IN (0,1))
    )
    """
    sql_for_inserting_job_ids = """
    INSERT OR IGNORE INTO job_processing_status (job_id) VALUES (?)
    """
    cursor.execute(sql_for_creating_table)
    conn.commit()
    cursor.executemany(sql_for_inserting_job_ids, convert_list_to_tuple(job_ids))
    conn.commit()
    conn.close()
    
def insert_job_details(job_details):
    sql_for_creation = """
    CREATE TABLE IF NOT EXISTS scraped_jobs (
    job_id TEXT UNIQUE PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    description TEXT NOT NULL,
    location TEXT,
    posted_date TEXT,
    application_url TEXT
    )
    """
    
    sql_for_insertion = """
    INSERT OR IGNORE INTO scraped_jobs VALUES (?,?,?,?,?,?,?)
    """
    cursor.execute(sql_for_creation)
    conn.commit()
    cursor.executemany(sql_for_insertion, job_details)
    conn.commit()
    conn.close()
