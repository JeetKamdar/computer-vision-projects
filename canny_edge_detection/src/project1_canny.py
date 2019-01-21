import sys
import numpy as np
import cv2

def convolution(image, gfilter, pad_i, pad_j):
    image_height, image_width = image.shape

    filter_height, filter_width = gfilter.shape
    
    #center of the filter
    center_i = int((filter_height - 1) / 2) 
    center_j = int((filter_width - 1) / 2)
    
    output = np.zeros((image_height, image_width))
    
    #padding is added to make sure the filter lies within the image
    for i in range(center_i + pad_i, image_height - (center_i + pad_i)):
        for j in range(center_j + pad_j, image_width - (center_j + pad_j)):
            sum = 0
            for k in range(-center_i, center_i+1):
                for l in range(-center_j, center_j+1):
                    x = image[i+k, j+l]
                    y = gfilter[center_i+k, center_j+l]
                    sum = sum + (x * y)
            output[i][j] = sum
    
    pad_i += center_i
    pad_j += center_j
    
    return output, pad_i, pad_j

def gaussian_filter(input_image):
    pad_i = 0
    pad_j = 0
    gfilter = np.array(
        [
            [1,1,2,2,2,1,1],
            [1,2,2,4,2,2,1],
            [2,2,4,8,4,2,2],
            [2,4,8,16,8,4,2],
            [2,2,4,8,4,2,2],
            [1,2,2,4,2,2,1],
            [1,1,2,2,2,1,1]
        ])
    gaussian_image, pad_i, pad_j = convolution(input_image, gfilter, pad_i, pad_j)
    
    #normalise the matrix generated in the previous step
    normalise = np.sum(gfilter)
    gaussian_image = gaussian_image / normalise

    cv2.imwrite('Lena-gaussian_image.jpg', gaussian_image)
    return gaussian_image, pad_i, pad_j
            
def gradient(gaussian_output, pad_i, pad_j):
    prewitt_x = np.array(
        [
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ])

    prewitt_y = np.array(
        [
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
        ])
    
    Gx, pad_i, pad_j = convolution(gaussian_output, prewitt_x, pad_i, pad_j)
    Gy, pad_i, pad_j = convolution(gaussian_output, prewitt_y, pad_i, pad_j)
    
    #calculate magnitude
    magnitude = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))

    #normalise
    Gx = np.abs(Gx) / 3
    Gy = np.abs(Gy) / 3
    magnitude = magnitude / np.sqrt(2)
    
    cv2.imwrite('Lena-prewittx_output.jpg', Gx)
    cv2.imwrite('Lena-prewitty_output.jpg', Gy)
    cv2.imwrite('Lena-magnitude_output.jpg', magnitude)
    
    return Gx, Gy, magnitude, pad_i, pad_j

def sector(Gx, Gy):
    #calculate angle
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    if theta < 0:
        theta += 360
    
    #calculate sector
    sector = 2
    if (theta > 337.5) or (theta >= 0 and theta <= 22.5) or (theta > 157.5 and theta <= 202.5):
        sector = 0
    elif (theta > 22.5 and theta <=67.5) or (theta > 202.5 and theta <= 247.5):
        sector = 1
    elif (theta > 67.5 and theta <=112.5) or (theta > 247.5 and theta <= 292.5):
        sector = 2
    elif (theta > 112.5 and theta <=157.5) or (theta > 292.5 and theta <= 337.5):
        sector = 3
    return sector

def calc_center_value(mag, i, j, theta, pad_i, pad_j):
    C = mag[i,j]

    #Calculate center value by comparing with relevant neighbours based on the sector
    if i < pad_i:
        return False
    elif j < pad_j:
        return False
    elif theta == 0:
         return mag[i, j-1] < C and mag[i, j+1] < C
    elif theta == 1:
        return mag[i-1, j+1] < C and mag[i+1, j-1] < C
    elif theta == 2:
        return mag[i-1, j] < C and mag[i+1, j] < C
    elif theta == 3:
        return mag[i-1, j-1] < C and mag[i+1, j+1] < C


def non_maxima_suppresion(magnitude, prewittx_output, prewitty_output, pad_i, pad_j):
    height, width = magnitude.shape
    nms = np.zeros((height, width))
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            Gx = prewittx_output[i,j]
            Gy = prewitty_output[i,j]
            theta = sector(Gx,Gy)
            if calc_center_value(magnitude, i, j, theta, pad_i, pad_j):
                nms[i,j] = magnitude[i,j]
            else:
                nms[i,j] = 0
    cv2.imwrite('Lena-nms_output.jpg', nms)
    return nms


def thresholding(nms_output, threshold):
    #convert to the nearest integer
    nms_integer = np.rint(nms_output)

    #sort to calculate the percentile value
    sorted_list = sorted(nms_integer[np.nonzero(nms_integer)])

    #find the value at index located at the threshold percentage in the list
    threshold_value = sorted_list[int(len(sorted_list) * (threshold / 100))] 

    final_image = np.zeros(nms_output.shape)
    for i in range(nms_output.shape[0]):
        for  j in range(nms_output.shape[1]):
            if nms_integer[i][j] >= threshold_value:
                final_image[i][j] = 255
    
    cv2.imwrite('Lena-P-tile' + str(100 - threshold)+'_output.jpg', final_image)


if __name__ == '__main__':
    input_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    input_image = input_image.astype(float)

    height, width = input_image.shape

    #keeps track of the undefined rows and columns
    pad_i = 0
    pad_j = 0

    #step 1
    gaussian_output, pad_i, pad_j = gaussian_filter(input_image)

    #step 2 and 3
    prewittx_output, prewitty_output, magnitude, pad_i, pad_j = gradient(gaussian_output, pad_i, pad_j)

    #step 4
    nms_output = non_maxima_suppresion(magnitude, prewittx_output, prewitty_output, pad_i, pad_j)

    #step 5
    P10_final_output = thresholding(nms_output, 90)
    P30_final_output = thresholding(nms_output, 70)
    P50_final_output = thresholding(nms_output, 50)

