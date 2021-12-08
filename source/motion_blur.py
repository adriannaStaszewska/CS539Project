import cv2
import numpy as np

# Blurring function
def blurImage(img, k):
    size_factor = 0.1 # Constant determining kernel size
    kernel_size = size_factor * k
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    
    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)
    
    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    # Normalize kernels
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    
    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    return vertical_mb, horizonal_mb

# Gets the highest speed limit sign in an annotation, -1 if not present
def getMaxSpeed(annotation_value):
    prefix = 'pl'
    speeds = [int(obj['category'][len(prefix):]) for obj in annotation_value['objects']
        if obj['category'].startswith(prefix)]
    return np.amax(speeds, initial=-1) # Return maximum speed found

# Gets mean and variance of all annotations
# The annotations passed should be the values of the json dictionary
def getSpeedDistribution(annotations):
    speeds = [getMaxSpeed(value) for (_, value) in annotations.items()] # Get all entries
    speeds = [speed for speed in speeds if speed > 0] # Filter out all negative/zero entries

    return np.mean(speeds), np.std(speeds)

# Gets the speed to use for an image
def getSpeed(annotation_value, mean, std):
    speed = getMaxSpeed(annotation_value)
    if speed <= 0: # No speed limit sign
        speed =  np.random.normal(mean, std)
    return speed # Return answer

"""
# DO NOT USE (old)
def test():
    imgs, _ = loadData('annotations.json')
    mean, std = getSpeedDistribution(imgs)
    tests = 100
    values = list(imgs.values())
    for _ in range(tests):
        annotation = np.random.choice(values)
        print(getSpeed(annotation, mean, std))
"""

# Orignal code in this file
def originalTest():
    img = cv2.imread('../../Downloads/stop.jpg')
    # Specify the kernel size.
    # The greater the size, the more the motion.
    k = 1200
    # Call function
    vertical_mb, horizontal_mb = blurImage(img, k)
    # Save the outputs.
    cv2.imwrite('image_vertical.jpg', vertical_mb)
    cv2.imwrite('image_horizontal.jpg', horizontal_mb)
