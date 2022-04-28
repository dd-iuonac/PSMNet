import os.path


class Logger:
    def __init__(self, path: str):
        self.path = path + os.path.sep + "log.txt"

    def write(self, message):
        with open(self.path, "a") as file:
            print(message + "\n")
            file.write(message + "\n")
