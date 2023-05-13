import logging
import time

import bchlib

import metrics
import neuroNetwork
import utils


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
    BCH_POLYNOMIAL = 8219
    BCH_BITS = 16
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    resultOfCheckDir = utils.checkDir()

    if not resultOfCheckDir:
        logging.debug("Directories created")
        return 0

    resMetrics = []
    resTimes = []
    logging.debug("Directories already exist")
    commands = utils.getAllCommands()

    for command in commands:
        startTime = time.time()
        transformedCommand = utils.transformText(command)
        encodedLine = utils.encodeToBCH(command, bch)
        
        neuroNetwork.runEncoder()
        neuroNetwork.runDecoder()

        decodedLine = utils.decodeFromBCH(encodedLine, bch)

        newLine = utils.neuroEmulation(command)

        resMetrics.append(metrics.neuroWorkMethod(command, newLine))
        resTimes.append(time.time() - startTime)

    utils.writeMetricsInFile(resMetrics, resTimes)


if __name__ == '__main__':

    main()
