import requests
import json
from app.ingestion.helpers import clean_job_desc, convert_to_ist, get_cookies, get_country_name  
from app.db.insert_data import insert_all_fetched_job_ids, insert_job_details 
import sqlite3

conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

class LinkedInJobsSearcher:
    def __init__(self):
        self.session = requests.Session()
        for cookie in get_cookies():
            self.session.cookies.set(cookie["name"], cookie["value"])

        self.session.headers = {
            "User-Agent": "Mozilla/5.0",
            "Csrf-Token": self.session.cookies["JSESSIONID"].strip('"'),
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Cookie": "; ".join([f"{key}={value}" for key, value in self.session.cookies.items()]),
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.linkedin.com/preload/"
        }
        self.search_url = "https://www.linkedin.com/voyager/api/voyagerJobsDashJobCards?decorationId=com.linkedin.voyager.dash.deco.jobs.search.JobSearchCardsCollection-220&count=100&q=jobSearch&query=(origin:JOB_SEARCH_PAGE_JOB_FILTER,keywords:%28%22data%20scientist%22%20OR%20%22machine%20learning%20engineer%22%20OR%20%22AI%20engineer%22%29,locationUnion:(geoId:91000000),selectedFilters:(experience:List(2,3),jobType:List(F),verifiedJob:List(true)),spellCorrectionEnabled:true)&start=500"
        
    def fetch_jobs(self):
        try:
            resp_jobs = self.session.get(url=self.search_url, headers=self.session.headers)
            resp_jobs.raise_for_status()
            if resp_jobs.status_code==200:
                jobs = resp_jobs.json()
                with open("jobs.json",'w') as file:
                    json.dump(jobs, file, indent=4)
                    
                total_jobs_fetched = jobs['data']['paging']['total']
                print("Total number of jobs available from URL: ", total_jobs_fetched)
                job_id_list = []
                for element in jobs['data']['elements']:
                    if "JobCard" in element["$type"]:
                        jobPostingCard = element["jobCardUnion"]["*jobPostingCard"]
                        job_id_list.append(jobPostingCard.split("(")[1].split(',')[0])
                     
                insert_all_fetched_job_ids(job_id_list)
        except Exception as e:
            raise SystemExit(e)

class LinkedInJobRetriever:
    def __init__(self):
        self.session = requests.Session()
        for cookie in get_cookies():
            self.session.cookies.set(cookie["name"], cookie["value"])
        self.session.headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": "; ".join([f"{key}={value}" for key,value in self.session.cookies.items()]),
            "Csrf-Token": self.session.cookies["JSESSIONID"].strip('"'),
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.linkedin.com/preload/"
        }
        self.search_url = "https://www.linkedin.com/voyager/api/jobs/jobPostings/{}?decorationId=com.linkedin.voyager.deco.jobs.web.shared.WebFullJobPosting-65"
        
    def JobDetails(self,jobId):
        try:
            resp_job_details = self.session.get(url=self.search_url.format(jobId), headers=self.session.headers)
            resp_job_details.raise_for_status()
            if resp_job_details.status_code==200:
                job_json = resp_job_details.json()
                with open("job_detail.json","w") as file:
                    json.dump(job_json, file)
                desc = clean_job_desc(job_json["data"]["description"]["text"])
                
                title = job_json["data"]["title"]
                location = job_json["data"]["formattedLocation"]
                country = get_country_name(location)
                jobApplicationUrl = ''
                if 'companyApplyUrl' in job_json["data"]["applyMethod"]:
                    jobApplicationUrl = job_json["data"]["applyMethod"]["companyApplyUrl"]
                elif 'easyApplyUrl' in job_json["data"]["applyMethod"]:
                    jobApplicationUrl = job_json["data"]["applyMethod"]["easyApplyUrl"]
                if 'companyName' in job_json['data']['companyDetails']:
                    companyName = job_json['data']['companyDetails']['companyName']
                elif job_json['included']:
                    for detail in job_json['included']:
                        
                        if detail['$type'] == "com.linkedin.voyager.organization.Company":
                            companyName = detail['name']
                originallyPostedAt = convert_to_ist(job_json["data"]['originalListedAt'])
                # print(title, companyName, desc[:80], location,originallyPostedAt,jobApplicationUrl)
                return jobId, title, companyName, desc, location,originallyPostedAt,jobApplicationUrl, country
                
        except Exception as e:
            print(f"Failed to fetch job {jobId}: {e}")
            return None
    
    def isJobExpired(self, jobId):
        try:
            resp_job_details = self.session.get(url=self.search_url.format(jobId), headers=self.session.headers)
            resp_job_details.raise_for_status()
            if resp_job_details.status_code==200:
                job_json = resp_job_details.json()
                with open("job_detail.json","w") as file:
                    json.dump(job_json, file)
                if job_json['data']['jobState'] == "CLOSED":
                    sql_query = 'DELETE FROM scraped_jobs WHERE job_id = ?'
                    cursor.execute(sql_query,(jobId,))
                    conn.commit()
        except Exception as e:
            raise SystemExit(e)

def insert_all_job_details():
    job_ids = get_all_jobs_from_db()
    job_retriever = LinkedInJobRetriever()
    job_details_for_all_ids = []
    for job_id in job_ids:
        result = job_retriever.JobDetails(jobId=job_id)
        if result is None:
            continue
        jobId, title, company, desc,location,posted_date, app_url,country = result
        job_details_for_all_ids.append((job_id, title, company, desc,location,posted_date, app_url, country))
    insert_job_details(job_details_for_all_ids)

def get_all_jobs_from_db():
    sql_query = "SELECT job_id FROM job_processing_status where is_scraped = 0"
    cursor.execute(sql_query)
    job_ids_list = cursor.fetchall()
    cleaned_list = []
    for job_id in job_ids_list:
        cleaned_list.append(job_id[0])

    return cleaned_list
insert_all_job_details()