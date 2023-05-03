def neuroWorkMethod(startLine, newLine):
    numberOfSameSymbols = 0
    minLengthOfComparedStrings = min(len(startLine), len(newLine))

    for i in range(minLengthOfComparedStrings):
        if ord(startLine[i]) == ord(newLine[i]):
            numberOfSameSymbols += 1

    accuracy = numberOfSameSymbols / max(len(startLine), len(newLine))

    return [accuracy, len(startLine), len(newLine), startLine, newLine]
