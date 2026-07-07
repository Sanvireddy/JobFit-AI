"""LinkedIn scraping: discover job ids, then fetch full job details.

Two clients wrap LinkedIn's internal Voyager API, authenticated with cookies
from a one-time Selenium login (see :mod:`app.ingestion.helpers`):

- :class:`LinkedInJobsSearcher` runs a job search and stores discovered ids.
- :class:`LinkedInJobRetriever` fetches full details for one job id.

Run with:  python -m app.ingestion.fetch_jobs
"""

import logging

import requests

from app.db import repository
from app.ingestion.helpers import (
    clean_job_desc,
    convert_to_ist,
    get_cookies,
    get_country_name,
)

logger = logging.getLogger(__name__)

# Voyager search query: ML/AI/DS keywords, worldwide (geoId 91000000),
# full-time, entry/associate experience levels, verified postings.
SEARCH_URL = (
    "https://www.linkedin.com/voyager/api/voyagerJobsDashJobCards"
    "?decorationId=com.linkedin.voyager.dash.deco.jobs.search.JobSearchCardsCollection-220"
    "&count=100&q=jobSearch"
    "&query=(origin:JOB_SEARCH_PAGE_JOB_FILTER,"
    "keywords:%28%22data%20scientist%22%20OR%20%22machine%20learning%20engineer%22%20OR%20%22AI%20engineer%22%29,"
    "locationUnion:(geoId:91000000),"
    "selectedFilters:(experience:List(2,3),jobType:List(F),verifiedJob:List(true)),"
    "spellCorrectionEnabled:true)&start=500"
)
JOB_DETAIL_URL = (
    "https://www.linkedin.com/voyager/api/jobs/jobPostings/{}"
    "?decorationId=com.linkedin.voyager.deco.jobs.web.shared.WebFullJobPosting-65"
)


def _build_session() -> requests.Session:
    """Create a requests session carrying LinkedIn auth cookies and headers."""
    session = requests.Session()
    for cookie in get_cookies():
        session.cookies.set(cookie["name"], cookie["value"])
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Csrf-Token": session.cookies["JSESSIONID"].strip('"'),
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Cookie": "; ".join(
                f"{key}={value}" for key, value in session.cookies.items()
            ),
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.linkedin.com/preload/",
        }
    )
    return session


class LinkedInJobsSearcher:
    """Runs the job search query and records discovered job ids."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or _build_session()

    def fetch_jobs(self) -> list[str]:
        response = self.session.get(SEARCH_URL)
        response.raise_for_status()
        payload = response.json()

        total = payload["data"]["paging"]["total"]
        logger.info("Total jobs available for this search: %s", total)

        job_ids = []
        for element in payload["data"]["elements"]:
            if "JobCard" in element["$type"]:
                card_ref = element["jobCardUnion"]["*jobPostingCard"]
                job_ids.append(card_ref.split("(")[1].split(",")[0])

        repository.insert_fetched_job_ids(job_ids)
        logger.info("Stored %d job ids", len(job_ids))
        return job_ids


class LinkedInJobRetriever:
    """Fetches full details for individual job postings."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or _build_session()

    def fetch_job_details(self, job_id: str):
        """Return a scraped_jobs row tuple for one job, or None on failure."""
        try:
            response = self.session.get(JOB_DETAIL_URL.format(job_id))
            response.raise_for_status()
            data = response.json()["data"]

            description = clean_job_desc(data["description"]["text"])
            title = data["title"]
            location = data["formattedLocation"]
            country = get_country_name(location)
            posted_at = convert_to_ist(data["originalListedAt"])

            apply_method = data.get("applyMethod", {})
            application_url = apply_method.get("companyApplyUrl") or apply_method.get(
                "easyApplyUrl", ""
            )

            company = data.get("companyDetails", {}).get("companyName")
            if company is None:
                # Fall back to the normalized Company entity in included[].
                for entity in response.json().get("included", []):
                    if entity["$type"] == "com.linkedin.voyager.organization.Company":
                        company = entity["name"]
                        break

            return (
                job_id,
                title,
                company,
                description,
                location,
                posted_at,
                application_url,
                country,
            )
        except Exception as exc:
            logger.warning("Failed to fetch job %s: %s", job_id, exc)
            return None

    def delete_if_expired(self, job_id: str) -> bool:
        """Check whether a posting is closed; delete it from the DB if so."""
        response = self.session.get(JOB_DETAIL_URL.format(job_id))
        response.raise_for_status()
        if response.json()["data"]["jobState"] == "CLOSED":
            repository.delete_job(job_id)
            logger.info("Deleted expired job %s", job_id)
            return True
        return False


def insert_all_job_details() -> None:
    """Fetch and store details for every discovered-but-unscraped job id."""
    job_ids = repository.get_unscraped_job_ids()
    logger.info("Fetching details for %d jobs...", len(job_ids))

    retriever = LinkedInJobRetriever()
    job_details = []
    for job_id in job_ids:
        result = retriever.fetch_job_details(job_id)
        if result is not None:
            job_details.append(result)

    repository.insert_job_details(job_details)
    # Drop roles we never want to apply to (internships, part-time).
    repository.delete_jobs_with_excluded_titles()
    logger.info("Stored details for %d jobs", len(job_details))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    insert_all_job_details()
