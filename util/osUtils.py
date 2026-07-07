import json


def readTriple(path, sep=None):
    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines = line.strip().split()
            yield lines


def readFile(path, sep=None):
    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines = line.strip().split()
            if len(lines) == 0:
                continue
            yield lines


def getJson(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dumpJson(obj, path):
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(obj, file)
