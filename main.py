import configparser
import logging
import time

import code_utils
import metrics
import neuroNetwork
import utils


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
    resultOfCheckDir = utils.checkDir()

    if not resultOfCheckDir:
        logging.debug("Directories created")
        return 0

    config = configparser.ConfigParser()
    config.read('settings/settings.ini')

    coder = code_utils.create_coder(config)
    resMetrics = []
    resTimes = []
    logging.debug("Directories already exist")
    commands = utils.getAllCommands()

    for command in commands:
        startTime = time.time()
        transformedCommand = utils.transformText(command)
        encodedLine = code_utils.encode(command, coder, config)
        
        neuroNetwork.runEncoder()
        neuroNetwork.runDecoder()

        decodedLine = code_utils.decode(encodedLine, coder, config)

        newLine = utils.neuroEmulation(command)

        resMetrics.append(metrics.neuroWorkMethod(command, newLine))
        resTimes.append(time.time() - startTime)

    utils.writeMetricsInFile(resMetrics, resTimes)


if __name__ == '__main__':
    main()
