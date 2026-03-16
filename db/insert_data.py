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
    job_ids = [job_detail[0] for job_detail in job_details]
    update_query = f"UPDATE job_processing_status SET is_scraped=1 WHERE job_id IN ({','.join(['?']*len(job_ids))})"
    cursor.execute(update_query,job_ids)
    conn.commit()
    delete_job_details()
    conn.close()

def delete_job_details():
    query = "DELETE FROM scraped_jobs where LOWER(title) LIKE '%intern%' OR LOWER(title) LIKE '%part%'"
    cursor.execute(query)
    conn.commit()
    conn.close()