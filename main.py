import logging
import time

import metrics
import neuroNetwork
import utils


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
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

        neuroNetwork.runEncoder()
        neuroNetwork.runDecoder()

        newLine = utils.neuroEmulation(command)

        resMetrics.append(metrics.neuroWorkMethod(command, newLine))
        resTimes.append(time.time() - startTime)

    utils.writeMetricsInFile(resMetrics, resTimes)


if __name__ == '__main__':

    main()
