import os

import requests

# dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = dir_path + "/file_storage/new.pdf"
#
# with open(dir_path, 'rb') as f:
#     print(f.read())
url = "http://localhost:8000/new.pdf"

r = requests.get(url=url)


file = open('new.pdf', 'wb')
file.write(r.content)