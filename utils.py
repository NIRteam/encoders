import pathlib
import random


def checkDir():
    try:
        createDir(False)
        return False
    except FileExistsError:
        print("Found directories")
        return True


def createDir(eraseError):
    pathlib.Path("data").mkdir(parents=True, exist_ok=eraseError)
    pathlib.Path("data/input").mkdir(parents=True, exist_ok=eraseError)
    pathlib.Path("data/output").mkdir(parents=True, exist_ok=eraseError)


def getAllCommands():
    with open("data/input/input.txt", "r") as inputFile:
        lines = inputFile.readlines()
        array = [line.strip() for line in lines]
        return array


def transformText(line):
    return line


def writeMetricsInFile(results, timeForRes):
    with open("numberOfRun.txt", "r") as numberOfRunFile:
        numberOfRun = numberOfRunFile.read()

    pathlib.Path("data/output/" + numberOfRun).mkdir(parents=True, exist_ok=True)

    with open("data/output/" + numberOfRun + "/metrics.txt", "w") as metricsFile:
        numberOfResult = 1
        for result in results:
            metricsMsg = ("\nresult № " + str(numberOfResult) + ":" +
                          "\n    accuracy: %.3f" % result[0] +
                          "\n    start line length: " + str(result[1]) +
                          "\n    result line length: " + str(result[2]) +
                          "\n    time for processing this command: " + str(timeForRes[numberOfResult - 1]) +
                          "\n    start line: " + result[3] +
                          "\n    new line: " + result[4] + "\n")
            metricsFile.write(metricsMsg)
            numberOfResult += 1
    with open("numberOfRun.txt", "w") as numberOfRunFile:
        numberOfRunFile.write(str(int(numberOfRun) + 1))


def neuroEmulation(line):
    result = ""
    randomObj = random.Random()
    for symbol in line:
        if randomObj.randint(a=1, b=50) == 42:
            symbol = '3'
        elif randomObj.randint(a=1, b=50) == 25:
            symbol = ''
        result += symbol
    return result
