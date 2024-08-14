import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(';')
            row = [int(v) for v in values if v]  
            data.append(row)
    return data

# image size
def get_image_size(data):
    max_x = max(row[0] for row in data)
    max_y = max(row[1] for row in data)
    return max_y + 1, max_y + 1 

# Create the "Total Count" image
def create_total_count_image(data, image_size):
    total_count = np.zeros(image_size, dtype=int)
    for row in data:
        x, y = row[:2]
        total_count[x, y] = len(row[2:])
    return total_count

# One Channel image
def create_channel_count_image(data, image_size, channel):
    channel_count = np.zeros(image_size, dtype=int)
    for row in data:
        x, y = row[:2]
        count = sum(1 for value in row[2:] if value == channel)
        channel_count[x, y] = count
    return channel_count

# Main function 
def process_data(filename, channel):
    data = read_data(filename)
    image_size = get_image_size(data)
    
    # images
    total_count_image = create_total_count_image(data, image_size)
    channel_count_image = create_channel_count_image(data, image_size, channel)
    
    return total_count_image, channel_count_image

#Total count image
def plot_total_count_image(total_count_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(total_count_image, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Total Count')
    plt.title('Total Count Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Image in a range of channels
def create_channel_range_count_image(data, image_size, channel_min, channel_max):
    range_count_image = np.zeros(image_size, dtype=int)
    
    for row in data:
        x, y = row[:2]
        count = sum(1 for value in row[2:] if channel_min <= value <= channel_max)
        range_count_image[x, y] += count
    
    return range_count_image

# Plot
def plot_channel_range_count_image(range_count_image, channel_min, channel_max):
    plt.figure(figsize=(8, 8))
    plt.imshow(range_count_image, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Count in Channel Range')
    plt.title(f'Channel Range Count Image ({channel_min}-{channel_max})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Usage
filename = 'FeTi_1000us.raw'  # Add file name
channel_min = 5000 # inferior channel
channel_max = 6000 #superior channel
channel = 6247  # Channel of interest

# Process data 
data = read_data(filename)
image_size = get_image_size(data)
total_count_image, channel_count_image = process_data(filename, channel)

# channel range count image
range_count_image = create_channel_range_count_image(data, image_size, channel_min, channel_max)


print("Total Count Image:\n", total_count_image)
print(f"Channel {channel} Count Image:\n", channel_count_image)

# Plot
plot_channel_range_count_image(range_count_image, channel_min, channel_max)
plot_total_count_image(channel_count_image)