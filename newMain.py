import logging
import time

import metrics
import neuroNetwork
import utils

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
    resultOfCheckDir = utils.checkDir()
    if resultOfCheckDir:
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

    else:
        logging.debug("Directories created")
