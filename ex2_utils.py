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
    if (len(k_size) > len(in_signal)):
        temp = k_size.copy()
        k_size = in_signal.copy()
        in_signal = temp.copy()
    shift = (len(k_size) - 1)
    # flip = np.ones(len(k_size))
    new = np.zeros(len(in_signal) + shift)
    # for i in range(len(k_size)):  # flipping the k_size vector
    #     flip[len(k_size) - 1 - i] = k_size[i]
    flip = np.flip(k_size)
    padded = in_signal.copy()
    for k in range(shift): # padding the vector with 0
        padded = np.insert(0, 0, padded)
        padded = np.append(0, padded)
        # print(padded)
    for i in range(len(padded) - shift):
        num = 0
        for k in range(len(k_size)):
            num += flip[k] * padded[i + k]
        # print(num)
        new[i] = num
    return new





def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # plt.imshow(in_image, cmap='gray')
    # plt.show()
    # shape_ker = kernel.shape
    # # print(shape_ker)
    # ker_row = shape_ker[0]
    # ker_col = shape_ker[1]
    # # print(shape_ker, ker_row, ker_col)
    # shape_img = in_image.shape
    # img_row = shape_img[0]
    # img_col = shape_img[1]
    # # print(shape_img, img_row, img_col)
    # new_img_blurred = np.zeros(shape_img)
    # new_img_check = np.zeros(shape_img)
    # # plt.imshow(new_img_blurred, cmap='gray')
    # # plt.show()
    # kernel=np.flip(kernel)
    # for i in range(img_row):
    #     for j in range(img_col):
    # #         #             print("i= ",i, "j= ",j)
    #         num = 0
    #         row = i - math.floor(ker_row / 2)
    #         col = j - math.floor(ker_col / 2)
    #
    #         if (row >= math.floor(ker_row / 2) and row < img_row - ker_row and col >= math.floor(ker_col / 2) and col < img_col - ker_col):
    #             # print("one of the above is true: this row is in [2,763]", row, "and this col is in [2,1147]", col)
    #             for k in range(ker_row):
    #                 for l in range(ker_col):
    #                     #                         print("i= ",i, "j= ",j, "k= ",k, "l ",l ,"gvhgvhjbj")
    #                     num = num + (kernel[k][l]) * (in_image[row + k][col + l])
    #             new_img_blurred[i][j] = num
    #             new_img_check[i][j] = 1
    #         else:
    #             # print("one of the above is true: this row is not in [2,763]", row, "and this col is not in [2,1147]",col)
    #             for k in range(ker_row):
    #                 for l in range(ker_col):
    #                     #                         print("i= ",i, "j= ",j, "k= ",k, "l ",l)
    #                     if (row + k >= 0 and row + k < img_row and col + j >= 0 and col + j < img_col):
    #                         num = num + (kernel[k][l]) * (in_image[row + k][col + l])
    #                     else:
    #                         r = row + k
    #                         c = col + l
    #                         if row + k < 0:
    #                             r = 0;
    #                         elif row + k >= img_row:
    #                             r = img_row - 1
    #                         if col + l < 0:
    #                             c = 0
    #                         elif col + l >= img_col:
    #                             c = img_col - 1
    #                         num = num + (kernel[k][l]) * (in_image[r][c])
    #             new_img_blurred[i][j] = num
    #             new_img_check[i][j] = 1
    # # print("new_img_blurred")
    # # plt.imshow(new_img_blurred, cmap='gray')
    # # plt.show()
    # return new_img_blurred


    shape_ker = kernel.shape
    ker_row = shape_ker[0]
    ker_col = shape_ker[1]

    shape_img = in_image.shape
    # print(shape_img)
    img_row = shape_img[0]
    img_col = shape_img[1]

    new_img_blurred = np.zeros(shape_img)
    # print(kernel)
    kernel = np.flip(kernel)
    # print(kernel)
    r_skip = math.floor(ker_row / 2)
    c_skip = math.floor(ker_col / 2)

    padded_image = cv2.copyMakeBorder(in_image, r_skip, r_skip, c_skip, c_skip, cv2.BORDER_REPLICATE, None, value=0)
    for i in range(img_row):
        for j in range(img_col):
            num = 0
            for k in range(ker_row):
                for l in range(ker_col):
                    num = num + (kernel[k][l]) * (padded_image[i+k][j+l])
            new_img_blurred[i][j] = num
    new_img_blurred=np.round(new_img_blurred)
    return new_img_blurred


def conv2D_REFLECT(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    shape_ker = kernel.shape
    ker_row = shape_ker[0]
    ker_col = shape_ker[1]

    shape_img = in_image.shape
    img_row = shape_img[0]
    img_col = shape_img[1]

    new_img_blurred = np.zeros(shape_img)
    r_skip = math.floor(ker_row / 2)
    c_skip = math.floor(ker_col / 2)

    padded_image = cv2.copyMakeBorder(in_image, r_skip, r_skip, c_skip, c_skip, cv2.BORDER_REFLECT_101, None, value=0)
    for i in range(img_row):
        for j in range(img_col):
            num = 0
            for k in range(ker_row):
                for l in range(ker_col):
                    num = num + (kernel[k][l]) * (padded_image[i+k][j+l])
            new_img_blurred[i][j] = num
    return new_img_blurred










def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    ker = np.array([[1, 0, -1]])
    kerT = np.transpose(ker)
    img_x = conv2D_REFLECT(in_image, ker)
    img_y = conv2D_REFLECT(in_image, kerT)
    MagG = np.sqrt((img_x * img_x ) + (img_y * img_y))
    dirG = np.arctan2(img_y, img_x).astype(np.float64)
    return dirG, MagG


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
