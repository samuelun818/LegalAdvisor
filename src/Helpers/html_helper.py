import requests
from bs4 import BeautifulSoup
import os

FILE_EXT = ".html"
headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

def get_urlcontent(url):
    # Send an HTTP GET request to the website
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    # Parse the HTML code using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def write_html(file_path, content):
    with open(file_path, 'wb') as file:
        file.write(content)
    return

def read_html(html_file):
    with  open(html_file, 'r', encoding="utf-8") as file:
        file_content = file.read()

    soup = BeautifulSoup(file_content, 'html.parser')
    text = soup.get_text()

    return text