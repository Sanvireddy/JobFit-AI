import sqlite3  

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
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql_for_creating_table)
        cursor.executemany(sql_for_inserting_job_ids, convert_list_to_tuple(job_ids))
        conn.commit()
    
def insert_job_details(job_details):
    sql_for_creation = """
    CREATE TABLE IF NOT EXISTS scraped_jobs (
    job_id TEXT UNIQUE PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    description TEXT NOT NULL,
    location TEXT,
    posted_date TEXT,
    application_url TEXT,
    country TEXT
    )
    """
    
    sql_for_insertion = """
    INSERT OR IGNORE INTO scraped_jobs VALUES (?,?,?,?,?,?,?,?)
    """
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql_for_creation)
        cursor.executemany(sql_for_insertion, job_details)
        conn.commit()

        job_ids = [job_detail[0] for job_detail in job_details]
        update_query = f"UPDATE job_processing_status SET is_scraped=1 WHERE job_id IN ({','.join(['?']*len(job_ids))})"
        cursor.execute(update_query,job_ids)
        conn.commit()

    delete_job_details()

def insert_jobs_metadata(job_ids_list):
    sql_for_creation = """
    CREATE TABLE IF NOT EXISTS job_metadata (
    job_id TEXT UNIQUE PRIMARY KEY,
    min_experience_years INTEGER,
    experience_requirement_text TEXT,
    requires_only_english BOOLEAN,
    language_requirement_text TEXT,
    requires_advanced_degree BOOLEAN,
    education_requirement_text TEXT,
    visa_sponsorship_available BOOLEAN,
    relocation_assistance_provided BOOLEAN,
    work_mode TEXT,
    relocation_evidence TEXT,
    FOREIGN KEY (job_id) REFERENCES scraped_jobs(job_id)
    )
    """

    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql_for_creation)
        conn.commit()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO job_metadata (
                job_id,
                min_experience_years,
                experience_requirement_text,
                requires_only_english,
                language_requirement_text,
                requires_advanced_degree,
                education_requirement_text,
                visa_sponsorship_available,
                relocation_assistance_provided,
                work_mode,
                relocation_evidence
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            job_ids_list
        )
        conn.commit()


def delete_job_details():
    query = "DELETE FROM scraped_jobs where LOWER(title) LIKE '%intern%' OR LOWER(title) LIKE '%part%'"
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
    
def get_data_from_scraped_jobs(columns_string):
    query = "SELECT "+columns_string+" FROM scraped_jobs WHERE is_processed = 0"
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
    return result
def mark_jobs_processed(job_ids):
    if not job_ids:
        return

    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    update_query = f"UPDATE scraped_jobs SET is_processed = 1 WHERE job_id IN ({','.join(['?'] * len(job_ids))})"
    cursor.execute(update_query, job_ids)
    conn.commit()
    conn.close()

def update_job_descriptions(job_list):
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.executemany("UPDATE scraped_jobs SET description = ? WHERE job_id = ?",job_list)
        conn.commit()
    
def update_extracted_prompt_attributes(job_list):
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.executemany("UPDATE scraped_jobs SET min_experience_years = ?, experience_requirement_text = ?, requires_only_english = ?, language_requirement_text = ?, requires_only_advanced_degree = ?, education_requirement_text = ?, is_processed = ? WHERE job_id = ?", job_list)
        conn.commit()

def get_latest_job_ids_with_null_faiss_index():
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, description FROM scraped_jobs where faiss_index IS NULL")
        result = cursor.fetchall()
    return result
def update_scraped_jobs_fields_from_prompt_response():
    sql_query = "SELECT job_id, description FROM scraped_jobs WHERE is_processed = 0 LIMIT 10"
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
    return result

def update_faiss_index(job_id, faiss_index):
    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE scraped_jobs SET faiss_index = ? WHERE job_id = ?", (faiss_index, job_id))
        conn.commit()

def get_jobs_with_faiss_index(faiss_index_list):
    if not faiss_index_list:
        return []

    with sqlite3.connect("jobs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT job_id, faiss_index FROM scraped_jobs WHERE faiss_index IN ({})".format(
                ','.join('?' * len(faiss_index_list))
            ),
            faiss_index_list,
        )
        result = cursor.fetchall()
    return result


def get_job_details_by_faiss_indices(faiss_indices):
    if not faiss_indices:
        return []

    placeholders = ",".join(["?"] * len(faiss_indices))
    sql_query = (
        "SELECT job_id, title, company, description, location, posted_date, "
        "application_url, country "
        "FROM scraped_jobs "
        f"WHERE faiss_index IN ({placeholders}) "
        "ORDER BY faiss_index"
    )

    with sqlite3.connect("jobs.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query, faiss_indices)
        result = cursor.fetchall()

    return [dict(row) for row in result]


def get_job_metadata_by_job_ids(job_ids):
    if not job_ids:
        return {}

    try:
        with sqlite3.connect("jobs.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(job_ids))
            cursor.execute(
                f"SELECT job_id, min_experience_years, requires_only_english, requires_advanced_degree, visa_sponsorship_available, relocation_assistance_provided, work_mode FROM job_metadata WHERE job_id IN ({placeholders})",
                job_ids,
            )
            results = cursor.fetchall()
            return {row["job_id"]: dict(row) for row in results}
    except Exception as e:
        print(f"Error retrieving job metadata: {e}")
        return {}