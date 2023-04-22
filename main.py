import logging
import random
import time


def getAllLinesFromFile():
    with open("data/input/input.txt", "r") as myFile:
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

        accuracy = numberOfSameSymbols / max(len(line), len(newLine))

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
    with open("metrics.txt", "w") as metricsFile:
        logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
        startTime = time.time()

        lines = getAllLinesFromFile()
        results = neuroWorkMethod(lines)
        logging.debug("\n---------------------------------------------------------------------\n"
                      + "                               results                               \n"
                      + "---------------------------------------------------------------------")

        numberOfResult = 1
        for result in results:
            metricsMsg = ("\nresult № " + str(numberOfResult) + ":" +
                          "\n    accuracy: %.3f" % result[0] +
                          "\n    start line length: " + str(result[1]) +
                          "\n    result line length: " + str(result[2]) +
                          "\n    time spent on processing: %.6f" % result[3] +
                          "\n    start line: " + result[4] +
                          "\n    new line: " + result[5] + "\n")
            logging.debug(metricsMsg)
            metricsFile.write(metricsMsg)
            numberOfResult += 1

        logging.debug("time for all processing: %.6f" % (time.time() - startTime))
