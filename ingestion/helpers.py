from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd

def get_username_password():
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
