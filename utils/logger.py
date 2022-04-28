import os.path
from datetime import datetime


class Logger:
    def __init__(self, path: str):
        self.path = path + os.path.sep + "log.txt"

    def write(self, message):
        text = f"[{datetime.now().strftime('%x - %X')}] {message}\n"

        with open(self.path, "a") as file:
            print(text)
            file.write(text)
