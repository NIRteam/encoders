import logging

import metrics
import neuroNetwork
import utils

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
    resultOfCheckDir = utils.checkDir()
    if resultOfCheckDir:
        resMetrics = []
        logging.debug("Directories already exist")
        commands = utils.getAllCommands()

        for command in commands:
            transformedCommand = utils.transformText(command)

            neuroNetwork.runEncoder()
            neuroNetwork.runDecoder()

            newLine = utils.neuroEmulation(command)

            resMetrics.append(metrics.neuroWorkMethod(command, newLine))

        utils.writeMetricsInFile(resMetrics)

    else:
        logging.debug("Directories created")
