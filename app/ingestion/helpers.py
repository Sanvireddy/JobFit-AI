"""Ingestion utilities: LinkedIn auth, text cleaning, and derived fields."""

import csv
import logging
import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pycountry
from geotext import GeoText

from app.config import ROOT_DIR

logger = logging.getLogger(__name__)

# Built once at import time instead of rebuilding the full pycountry set on
# every get_country_name() call.
_COUNTRY_NAMES = frozenset(country.name for country in pycountry.countries)

LOGINS_CSV_PATH = ROOT_DIR / "logins.csv"


def get_username_password():
    """Return (email, password) for LinkedIn login.

    Prefers the LINKEDIN_EMAIL / LINKEDIN_PASSWORD environment variables so
    credentials never need to live in a file. Falls back to the legacy
    logins.csv (gitignored) for existing local setups.
    """
    env_email = os.environ.get("LINKEDIN_EMAIL")
    env_password = os.environ.get("LINKEDIN_PASSWORD")
    if env_email and env_password:
        return env_email, env_password

    with open(LOGINS_CSV_PATH, newline="") as file:
        row = next(csv.DictReader(file))
    return row["emails"], row["passwords"]


def get_cookies():
    """Log into LinkedIn with Selenium and return the session cookies."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    driver = webdriver.Chrome()
    try:
        driver.get("https://www.linkedin.com/login")
        assert "Linked" in driver.title
        username, password = get_username_password()
        driver.find_element(By.ID, "username").send_keys(username)
        driver.find_element(By.ID, "password").send_keys(password)
        time.sleep(1)
        driver.find_element(By.XPATH, '//button[@aria-label="Sign in"]').click()
        time.sleep(1)
        cookies = driver.get_cookies()
        time.sleep(2)
    finally:
        driver.quit()
    logger.info("Successfully logged in to LinkedIn")
    return cookies


def clean_job_desc(job_desc: str) -> str:
    job_desc = job_desc.strip()
    job_desc = re.sub(r"\n+", "\n", job_desc)
    job_desc = re.sub(r"\s+", " ", job_desc)
    return job_desc


def convert_to_ist(timestamp_ms: int) -> datetime:
    """Convert an epoch-milliseconds timestamp to IST."""
    utc_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return utc_time.astimezone(ZoneInfo("Asia/Kolkata"))


def get_country_name(location: str) -> str:
    places = GeoText(text=location)
    extracted = [c for c in places.countries if c in _COUNTRY_NAMES]
    return " ".join(extracted)


def compute_is_only_english_required(languages):
    """Derive the (requires_only_english, evidence) pair from language requirements.

    Returns (None, None) when nothing is explicitly required, (True, evidence)
    when English is the only required language, and (False, evidence) when any
    non-English language is mandatory.
    """
    if languages is None:
        return None, None

    required = [lang for lang in languages if lang.requirement == "required"]
    if not required:
        return None, None

    evidence = " | ".join(lang.evidence for lang in required)
    required_names = {lang.language.lower() for lang in required}
    return required_names == {"english"}, evidence
