
def getAllLinesFromFile():
    with open("file.txt", "r") as myFile:
        lines = myFile.readlines()
        array = [line.strip() for line in lines]
        return array


def neuroWorkMethod():
    pass


if __name__ == '__main__':
    print(getAllLinesFromFile())
