import requests
import json

class LinkedInJobsSearcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": "Cookie",
            "Csrf-Token": "ajax:0433998784187860844",
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.linkedin.com/preload/"
        }
        self.search_url = "https://www.linkedin.com/voyager/api/voyagerJobsDashJobCards?decorationId=com.linkedin.voyager.dash.deco.jobs.search.JobSearchCardsCollection-220&count=10&q=jobSearch&query=(origin:JOB_SEARCH_PAGE_JOB_FILTER,keywords:data%20scientist,locationUnion:(geoId:102713980),selectedFilters:(experience:List(2,3)),spellCorrectionEnabled:true)&start=100"
        
    def fetch_jobs(self):
        try:
            resp_jobs = self.session.get(url=self.url, headers=self.headers)
            resp_jobs.raise_for_status()
            if resp_jobs.status_code==200:
                jobs_json = resp_jobs.json()
                with open("jobs.json",'w') as file:
                    json.dump(jobs_json, file, indent=4)
                return jobs_json
        except Exception as e:
            raise SystemExit(e)

    def parse_json_job_list(self,jobs):
        total_jobs_fetched = jobs['data']['paging']['total']
        print("Total number of jobs available from URL: ", total_jobs_fetched)
        job_id_list = []
        for element in jobs['data']['elements']:
            if "JobCard" in element["$type"]:
                jobPostingCard = element["jobCardUnion"]["*jobPostingCard"]
                job_id_list.append(jobPostingCard.split("(")[1].split(',')[0])
        return job_id_list    

class LinkedInJobRetriever:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": "Cookie",
            "Csrf-Token": "ajax:0433998784187860844",
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.linkedin.com/preload/"
        }
        self.search_url = "https://www.linkedin.com/voyager/api/jobs/jobPostings/{}?decorationId=com.linkedin.voyager.deco.jobs.web.shared.WebFullJobPosting-65"
        
    def JobDetails(self,jobId):
        try:
            
            resp_job_details = self.session.get(url=self.url.format(jobId), headers=self.headers)
            resp_job_details.raise_for_status()
            if resp_job_details.status_code==200:
                resp_job_details_json = resp_job_details.json()
                with open("job_detail.json","w") as file:
                    json.dump(resp_job_details_json, file)
                
        except Exception as e:
            raise SystemExit(e)

    def get_job_details(job_json):
        desc = job_json["description"]["text"]
        title = job_json["title"]
        location = job_json["formattedLocation"]
        jobApplicationUrl = job_json["jobPostingUr;"]
        return desc, title, location, jobApplicationUrl
    
