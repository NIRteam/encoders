import random
import time


def getAllLinesFromFile():
    with open("file.txt", "r") as myFile:
        lines = myFile.readlines()
        array = [line.strip() for line in lines]
        return array


def neuroWorkMethod(lines):
    results = []
    for line in lines:
        startTime = time.time()
        numberOfSameSymbols = 0

        newLine = neuroEmulation(line)
        minLengthOfComparedStrings = min(len(line), len(newLine))

        for i in range(minLengthOfComparedStrings):
            if ord(line[i]) == ord(newLine[i]):
                numberOfSameSymbols += 1

        accuracy = numberOfSameSymbols/max(len(line), len(newLine))

        results.append([accuracy, len(line), len(newLine), (time.time() - startTime), line, newLine])
    return results


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


if __name__ == '__main__':
    startTime = time.time()

    lines = getAllLinesFromFile()
    results = neuroWorkMethod(lines)
    print("---------------------------------------------------------------------\n"
          + "                               results                               \n"
          + "---------------------------------------------------------------------")

    numberOfResult = 1
    for result in results:
        print("result №" + str(numberOfResult) + ":",
              "\n    accuracy: %.3f" % result[0], "\n    start line length: ", result[1],
              "\n    result line length: ", result[2], "\n    time spent on processing: %.6f" % result[3],
              "\n    start line: ", result[4], "\n    new line: ", result[5], "\n")
        numberOfResult += 1

    print("time for all processing: %.6f" % (time.time() - startTime))
