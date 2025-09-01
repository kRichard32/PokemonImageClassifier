import os
from functools import cached_property
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl, urlparse
from requests_toolbelt.multipart import decoder
from pathlib import Path
import json
import re

import OnnxClient


class WebRequestHandler(BaseHTTPRequestHandler):
    @cached_property
    def url(self):
        return urlparse(self.path)

    @cached_property
    def query_data(self):
        return dict(parse_qsl(self.url.query))

    @cached_property
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    @cached_property
    def form_data(self):
        return dict(parse_qsl(self.post_data.decode("utf-8")))

    @cached_property
    def cookies(self):
        return SimpleCookie(self.headers.get("Cookie"))

    # def get_response(self):
    #     return json.dumps(
    #         {
    #             "path": self.url.path,
    #             "query_data": self.query_data,
    #             "post_data": self.post_data.decode("utf-8"),
    #             "form_data": self.form_data,
    #             "cookies": {
    #                 name: cookie.value
    #                 for name, cookie in self.cookies.items()
    #             },
    #         }
    #     )
    def get_data(self, path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path + "/file_storage/" + path
        with open(dir_path, 'rb') as f:
            return f.read()
    def grab_data(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/pdf')
        self.send_header('Content-Disposition', 'attachment; filename="file.pdf"')
        self.end_headers()
        self.wfile.write(self.get_data(self.path))

    def classify(self):
        data_path = self.query_data["filePath"]
        image_path = "file_storage/" + data_path
        answer = OnnxClient.run_classification(image_path=image_path)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(answer.encode('ascii'))
    def do_GET(self):
        if self.path.startswith("/classify"):
            self.classify()
        else:
            self.grab_data()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)

        content_type = self.headers['Content-Type']

        specific_content_type = content_type.split(";")[0]
        if specific_content_type == 'application/octet-stream':
            filename = self.query_data['filename']
            file_bytes = data
        elif specific_content_type == 'multipart/form-data':
            multipart_string = data
            multipart_data = decoder.MultipartDecoder(multipart_string, content_type)

            part = multipart_data.parts[0]
            file_bytes = part.content
            filename = (part.headers[b'Content-Disposition'].split(b';')[2].split(b'=')[1]).decode("utf-8")[1:-1]
        else:
            answer = "program does not support content type %s" % specific_content_type
            print(answer)
            self.send_response(415)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(answer.encode("utf-8"))
            return
        new_path = self.get_available_path(filename)
        file = open(f'{os.path.dirname(os.path.realpath(__file__))}/file_storage/{new_path}', 'wb')
        file.write(file_bytes)
        file.close()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(new_path.encode("utf-8"))
    def get_available_path(self, filename):
        stem, suffix = filename.split(".")
        suffix = "." + suffix
        path = os.path.dirname(os.path.realpath(__file__)) + "/file_storage"
        files = list(Path(path).iterdir())
        matches = [item for item in files if re.match(f"{stem}_?[0-9]*$",item.stem) and item.suffix == suffix]
        if len(matches) == 0:
            return filename
        return f'{stem}_{len(matches)}{suffix}'

if __name__ == "__main__":
    print("Starting Server")
    server = HTTPServer(("0.0.0.0", 8000), WebRequestHandler)
    server.serve_forever()
    # curl http://localhost:8000/v2/health/ready
    # docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 -v ${PWD}:/models nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model pokemon_prediction_model
    # docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 -v ${PWD}/../ImageClassifier:/models nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model pokemon_prediction_model