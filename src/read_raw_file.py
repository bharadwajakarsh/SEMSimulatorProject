import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_raw_data(fileName):
    rawData = []
    with open(fileName, 'r') as file:
        for line in file:
            values = line.strip().split(';')
            row = [int(v) for v in values if v]
            rawData.append(row)
    return rawData


def get_image_size(rawData):
    maxX = max(row[0] for row in rawData)
    maxY = max(row[1] for row in rawData)
    return maxY + 1, maxX + 1


def create_total_count_image(rawData, imageSize):
    totalCount = np.zeros(imageSize, dtype=int)
    for row in rawData:
        x, y = row[:2]
        totalCount[x, y] = len(row[2:])
    return totalCount


def create_channel_count_image(rawData, imageSize, channelNumber):
    channelCount = np.zeros(imageSize, dtype=int)
    for row in rawData:
        x, y = row[:2]
        count = sum(1 for value in row[2:] if value == channelNumber)
        channelCount[x, y] = count
    return channelCount


def process_data(fileName, channelNumber):
    rawData = read_raw_data(fileName)
    imageSize = get_image_size(rawData)

    # images
    totalCountImage = create_total_count_image(rawData, imageSize)
    channelCountImage = create_channel_count_image(rawData, imageSize, channelNumber)

    return totalCountImage, channelCountImage


def plot_total_count_image(totalCountImage):
    plt.figure(figsize=(8, 8))
    plt.imshow(totalCountImage, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Total Count')
    plt.title('Total Count Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


# Image in a range of channels
def create_channel_range_count_image(rawData, imageSize, channelLowerBound, channelUpperBound):
    rangeCountImage = np.zeros(imageSize, dtype=int)

    for row in rawData:
        x, y = row[:2]
        count = sum(1 for value in row[2:] if channelLowerBound <= value <= channelUpperBound)
        rangeCountImage[x, y] += count

    return rangeCountImage


# Plot
def plot_channel_range_count_image(rangeCountImage, channelLowerBound, channelUpperBound):
    plt.figure(figsize=(8, 8))
    plt.imshow(rangeCountImage, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Count in Channel Range')
    plt.title(f'Channel Range Count Image ({channelLowerBound}-{channelUpperBound})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
