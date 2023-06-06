import cv2
import numpy as np
from scipy.spatial.distance import hamming
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_metric(image1, image2):
    image1_vector = np.array(image1).ravel()
    image2_vector = np.array(image2).ravel()

    similarity_score = cosine_similarity(
        [image1_vector], [image2_vector])[0][0]

    return similarity_score


def hamming_distance_metric(image1, image2):
    return hamming(image1.flatten(), image2.flatten())


def mse_metric(image1, image2):
    mse = np.mean((image1 - image2) ** 2)

    return mse


def ssim(image1, image2):
    """TODO: Needs checking! Probably wrong!"""
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score = structural_similarity(image1, image2, data_range=image2.max() - image2.min())

    return score


def cor_pirson(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    corr = ((data1 - mean1) * (data2 - mean2)).mean() / (std1 * std2)
    # corr = ((data1 * data2).mean() - mean1 * mean2) / (std1 * std2)
    return corr


def generate_metrics(test_inputs, test_outputs, counter):
    test_input_img = test_inputs[counter].permute(1, 2, 0).detach().cpu().numpy()
    test_output_img = test_outputs[counter].permute(1, 2, 0).detach().cpu().numpy()

    mse_score = mse_metric(test_input_img, test_output_img)
    ssim_score = ssim(test_input_img, test_output_img)
    cosine_similarity_score = cosine_similarity_metric(test_input_img, test_output_img)
    hamming_distance_score = hamming_distance_metric(test_input_img, test_output_img)
    cor_pirson_score = cor_pirson(test_input_img, test_output_img)

    return mse_score, ssim_score, cosine_similarity_score, hamming_distance_score, cor_pirson_score, test_output_img


#  old metrics method
def neuroWorkMethod(startLine, newLine):
    numberOfSameSymbols = 0
    minLengthOfComparedStrings = min(len(startLine), len(newLine))

    for i in range(minLengthOfComparedStrings):
        if ord(startLine[i]) == ord(newLine[i]):
            numberOfSameSymbols += 1

    accuracy = numberOfSameSymbols / max(len(startLine), len(newLine))

    return [accuracy, len(startLine), len(newLine), startLine, newLine]
