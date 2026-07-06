import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pycountry
from geotext import GeoText

# Built once at import time instead of rebuilding the full pycountry set on every
# get_country_name() call.
_COUNTRY_NAMES = frozenset(country.name for country in pycountry.countries)


def get_username_password():
    """Return (email, password) for LinkedIn login.

    Prefers the LINKEDIN_EMAIL / LINKEDIN_PASSWORD environment variables so that
    credentials never need to live in a tracked file. Falls back to the legacy
    logins.csv for backward compatibility with existing local setups.
    """
    env_email = os.environ.get("LINKEDIN_EMAIL")
    env_password = os.environ.get("LINKEDIN_PASSWORD")
    if env_email and env_password:
        return env_email, env_password

    df = pd.read_csv("logins.csv")
    emails = df['emails']
    passwords = df['passwords']
    return emails[0], passwords[0]
    
def get_cookies():
    driver = webdriver.Chrome()

    driver.get('https://www.linkedin.com/login')
    assert 'Linked' in driver.title
    username, password = get_username_password()
    username_element=driver.find_element(By.ID, 'username')
    password_element=driver.find_element(By.ID,"password")
    username_element.click()
    username_element.send_keys(username)
    password_element.send_keys(password)
    time.sleep(1)
    driver.find_element(By.XPATH, '//button[@aria-label="Sign in"]').click()
    time.sleep(1)
    cookies = driver.get_cookies()
    time.sleep(2)
    driver.quit()
    print("Successfully Logged in")
    return cookies

def clean_job_desc(job_desc):
    job_desc = job_desc.strip()
    job_desc = re.sub(r'\n+','\n',job_desc)
    job_desc = re.sub(r'\s+',' ',job_desc)
    return job_desc


def convert_to_ist(time):
    utc_time = datetime.fromtimestamp(time/1000,tz=timezone.utc)
    return utc_time.astimezone(ZoneInfo("Asia/Kolkata"))

def get_country_name(location):
    places = GeoText(text=location)
    extracted_countries = [c for c in places.countries if c in _COUNTRY_NAMES]
    return ' '.join(extracted_countries)

def compute_is_only_english_required(
    languages
):
    if languages is None:
        return None, None

    required_languages = [
        lang
        for lang in languages
        if lang.requirement == "required"
    ]

    if not required_languages:
        return None, None

    required_language_names = {
        lang.language.lower()
        for lang in required_languages
    }

    if required_language_names == {"english"}:

        evidence = " | ".join(
            lang.evidence
            for lang in required_languages
        )

        return True, evidence

    return False, " | ".join(
        lang.evidence
        for lang in required_languages
    )
def compare_with_date(curr_date, year, month, date):
    cutoff = datetime(year, month, date, tzinfo=ZoneInfo("Asia/Kolkata"))
    if curr_date > cutoff:
        return True
    return False

def is_maters_or_phd_required(higher_education_requirement):
    if higher_education_requirement in ["masters", "phd"]:
        return True
    return False
