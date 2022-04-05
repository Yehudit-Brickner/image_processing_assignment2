import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    flip = np.ones(len(k_size))
    new = np.zeros(len(in_signal) + (len(k_size) - 1))
    for i in range(len(k_size)):  # fliiping the k_size vector
        flip[len(k_size) - 1 - i] = k_size[i]
    #     print("filip " ,flip)
    #     print("k-size ",k_size)
    for i in range((0 - len(k_size) + 1), len(in_signal)):
        #         print(i)
        if (i < 0):  # if we are at the beginig and need to add zeroes b4 the start of the in signal
            part = np.zeros(len(k_size))
            j = 0
            for k in range(-1 * i):  # fill in 0 at begining
                part[j] = 0;
                j = j + 1
            l = len(k_size) - j
            for k in range(j, len(k_size)):  # fill in numbers from in signal
                m = i + len(k_size) - l
                part[j] = in_signal[i + len(k_size) - l]
                j = j + 1
                l = l - 1;
            num = 0;
            #             print(part)
            for k in range(len(k_size)):  # find the new num
                num += part[k] * flip[k]
            new[i + len(k_size) - 1] = num
        #             print(num)
        #             print(new)
        #             print()
        elif (i > (
                len(in_signal) - len(k_size))):  # if we are at the end and need to add zeroes at the of the in signal
            part = np.zeros(len(k_size))
            j = 0
            m = i
            for k in range(i, len(in_signal)):  # fill in numbers from in signal
                part[j] = in_signal[m]
                m = m + 1
                j = j + 1
            # i dont need to add zeros at the end because the array was defalted to zeros
            num = 0;
            #             print(part)
            for k in range(len(k_size)):  # find the new num
                num += part[k] * flip[k]
            new[i + len(k_size) - 1] = num
        #             print(num)
        #             print(new)
        #             print()
        else:
            j = 0
            part = np.zeros(len(k_size))
            for k in range(len(k_size)):
                part[j] = in_signal[i + k]
                j = j + 1
            #             print(part)
            num = 0
            for k in range(len(k_size)):  # find the new num
                num += part[k] * flip[k]
            new[i + len(k_size) - 1] = num
    #             print(num)
    #             print(new)
    #             print()
    #     print(new)
    return new


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    plt.imshow(in_image, cmap='gray')
    plt.show()
    shape_ker = kernel.shape
    print(shape_ker)
    ker_row = shape_ker[0]
    ker_col = shape_ker[1]
    print(shape_ker, ker_row, ker_col)
    shape_img = in_image.shape
    img_row = shape_img[0]
    img_col = shape_img[1]
    print(shape_img, img_row, img_col)
    new_img_blurred = np.zeros(shape_img)
    new_img_check = np.zeros(shape_img)
    # plt.imshow(new_img_blurred, cmap='gray')
    # plt.show()

    for i in range(img_row):
        for j in range(img_col):
            #             print("i= ",i, "j= ",j)
            num = 0
            row = i - math.floor(ker_row / 2)
            col = j - math.floor(ker_col / 2)

            if (row >= math.floor(ker_row / 2) and row < img_row - ker_row and col >= math.floor(ker_col / 2) and col < img_col - ker_col):
                # print("one of the above is true: this row is in [2,763]", row, "and this col is in [2,1147]", col)
                for k in range(ker_row):
                    for l in range(ker_col):
                        #                         print("i= ",i, "j= ",j, "k= ",k, "l ",l ,"gvhgvhjbj")
                        num = num + (kernel[k][l]) * (in_image[row + k][col + l])
                new_img_blurred[i][j] = num
                new_img_check[i][j] = 1
            else:
                # print("one of the above is true: this row is not in [2,763]", row, "and this col is not in [2,1147]",col)
                for k in range(ker_row):
                    for l in range(ker_col):
                        #                         print("i= ",i, "j= ",j, "k= ",k, "l ",l)
                        if (row + k >= 0 and row + k < img_row and col + j >= 0 and col + j < img_col):
                            num = num + (kernel[k][l]) * (in_image[row + k][col + l])
                        else:
                            r = row + k
                            c = col + l
                            if row + k < 0:
                                r = 0;
                            elif row + k >= img_row:
                                r = img_row - 1
                            if col + l < 0:
                                c = 0
                            elif col + l >= img_col:
                                c = img_col - 1
                            num = num + (kernel[k][l]) * (in_image[r][c])
                new_img_blurred[i][j] = num
                new_img_check[i][j] = 1
    # print("new_img_blurred")
    # plt.imshow(new_img_blurred, cmap='gray')
    # plt.show()


    return new_img_blurred




def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    return


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
