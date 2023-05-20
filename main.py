import configparser
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import code_utils
import metrics
import neuroNetwork
import utils


def old_main():
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


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")
    resultOfCheckDir = utils.checkDir()

    if not resultOfCheckDir:
        logging.debug("Directories created")
        return 0

    config = configparser.ConfigParser()
    config.read('settings/settings.ini')

    # задание параметров нейронной сети
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    transform = transforms.Compose([
        transforms.Resize(config.getint('transform', 'x'), config.getint('transform', 'y')),
        transforms.ToTensor()
    ])

    for i in range(1, len(next(os.walk('data/input'))[1])):
        model = neuroNetwork.Autoencoder()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        #  обучение нейронной сети
        if config.getint('run', 'mod') in [0, 1]:
            model = model_learn(model, device, criterion, transform, optimizer)

        #  прогон тестов нейронной сети
        if config.getint('run', 'mod') in [0, 2]:
            model_test(model, device, criterion, transform)


def model_learn(model, device, criterion, transform, optimizer):
    dataset = torchvision.datasets.ImageFolder(root='drive/MyDrive/Colab Notebooks/data/input', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for epoch in range(10):
        running_loss = 0.0
        number_img = 0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = neuroNetwork.getOutputs(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), f'drive/MyDrive/data/веса/autoencoder, 512, {epoch + 101} эпох.pth')
    torch.save(model.state_dict(), 'drive/MyDrive/Colab Notebooks/data/веса/autoencoder.pth')

    return model


def model_test(model, device, criterion, transform):
    model.eval()
    test_dataset = torchvision.datasets.ImageFolder(root="drive/MyDrive/Colab Notebooks/data/test", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    mse_scores, ssim_scores, cosine_similarity_scores, hamming_distance_scores, cor_pirson_scores = [], [], [], [], []

    with torch.no_grad():
        number_img = 0
        test_loss = 0.0
        for i, test_data in enumerate(test_dataloader, 0):
            test_inputs, _ = test_data
            test_inputs = test_inputs.to(device)
            test_outputs = neuroNetwork.getOutputs(test_inputs)
            test_loss += criterion(test_outputs, test_inputs).item()
            for j in range(len(test_inputs)):
                number_img += 1

                counted_metrics = metrics.generate_metrics(test_inputs, test_outputs, j)

                # Добавление метрик в списки
                mse_scores.append(counted_metrics[0])
                ssim_scores.append(counted_metrics[1])
                cosine_similarity_scores.append(counted_metrics[2])
                hamming_distance_scores.append(counted_metrics[3])
                cor_pirson_scores.append(counted_metrics[4])

                Image.fromarray((counted_metrics[5] * 255).astype('uint8')).save(
                    f'drive/MyDrive/Colab Notebooks/data/output/test_output_{number_img}.jpg'
                )

    test_mse = np.mean(mse_scores)
    test_ssim = np.mean(ssim_scores)
    test_cosine_similarity = np.mean(cosine_similarity_scores)
    test_hamming_distance = np.mean(hamming_distance_scores)
    test_cor_pirson = np.mean(cor_pirson_scores)

    # Запись метрик в файл
    with open('metrics.txt', 'w') as f:
        f.write(f'Test MSE: {test_mse}\n')
        f.write(f'Test SSIM: {test_ssim}\n')
        f.write(f'Test cosine similarity: {test_cosine_similarity}\n')
        f.write(f'Test hamming_distance: {test_hamming_distance}\n')
        f.write(f'Test cor pirson: {test_cor_pirson}\n')

        # Запись всех метрик в текстовый файл
        all_metrics = np.array([test_mse, test_ssim, test_cosine_similarity, test_hamming_distance, test_cor_pirson])
        np.savetxt(f, all_metrics)


if __name__ == '__main__':
    main()
