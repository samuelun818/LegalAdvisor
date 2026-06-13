from bs4 import BeautifulSoup
import os

FILE_EXT = ".html"

def writ_html(file_path, content):
    with open(file_path, 'wb') as file:
        file.write(content)
    return

def read_html(html_file):
    with  open(html_file, 'r', encoding="utf-8") as file:
        file_content = file.read()

    soup = BeautifulSoup(file_content, 'html.parser')
    text = soup.get_text()

    return text