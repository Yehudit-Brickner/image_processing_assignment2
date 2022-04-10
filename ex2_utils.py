import math
import numpy as np
import cv2
import matplotlib.pyplot as plt



def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 328601018


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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

    shape_ker = kernel.shape
    ker_row = shape_ker[0]
    ker_col = shape_ker[1]

    shape_img = in_image.shape
    img_row = shape_img[0]
    img_col = shape_img[1]

    new_img_blurred = np.zeros(shape_img)
    kernel = np.flip(kernel)
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
    # smooth=np.array([[1,2,1],[2,4,2],[1,2,1]])
    # # smooth=smooth/np.sum(smooth)
    smooth=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    smooth = np.array([[0,0,1,2,1,0,0], [0,3,13,22,13,3,0], [1,13,59,97,59,13,1], [2,22,97,159,97,22,2],[1,13,59,97,59,13,1],[0,3,13,22,13,3,0],[0,0,1,2,1,0,0]])
    # # smooth = smooth / np.sum(smooth)
    img_smothed=conv2D(img,smooth)

    lap_filter=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    filterd=conv2D(img_smothed,lap_filter)
    # plt.imshow(filterd)
    # plt.show()
    shape=filterd.shape
    row=shape[0]
    col=shape[1]
    new_img=np.zeros(shape)
    #we will not include the edges so that we can find the other edges easier
    for i in range (1,row-1):
        for j in range(1,col-1):
            center =filterd[i][j]
            up= filterd[i-1][j]
            down =filterd[i+1][j]
            left= filterd[i][j-1]
            right=filterd[i][j+1]
            if(center==0):
                if(right*left<0):
                    new_img[i][j]=1
                if(up*down<0):
                    new_img[i][j] = 1
            elif (center*up<0):
                new_img[i][j] = 1
                new_img[i-1][j] = 1
            elif (center * down < 0):
                new_img[i][j] = 1
                new_img[i+1][j] = 1
            elif (center*left<0):
                new_img[i][j] = 1
                new_img[i][j-1] = 1
            elif (center*right<0):
                new_img[i][j] = 1
                new_img[i][j+1] = 1
    # plt.imshow(new_img)
    # plt.show()

    return new_img


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

    ker=np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    # plt.imshow(img)
    # plt.show()
    img=conv2D(img,ker)
    # plt.imshow(img)
    # plt.show()
    edge_img = edgeDetectionZeroCrossingLOG(img)
    # plt.imshow(edge_img)
    # plt.show()
    shape = edge_img.shape
    row = shape[0]
    col = shape[1]
    pi = math.pi
    # if(row<150 and col<150):
    #     rowarr = np.arange(0, row,1).astype(int)
    #     colarr = np.arange(0, col,1).astype(int)
    # elif(row<300 and col<300):
    #     rowarr = np.arange(0, row,3 ).astype(int)
    #     colarr = np.arange(0, col, 3).astype(int)
    # elif (row < 500 and col < 500):
    #     rowarr = np.arange(0, row, 5).astype(int)
    #     colarr = np.arange(0, col, 5).astype(int)
    # else :
    #     rowarr = np.arange(0, row, 10).astype(int)
    #     colarr = np.arange(0, col, 10).astype(int)
    if (row <=400 and col <=400):
        rowarr = np.arange(0, row, 1).astype(int)
        colarr = np.arange(0, col, 1).astype(int)
    elif (row <=600 and col <=600):
        rowarr = np.arange(0, row, 2).astype(int)
        colarr = np.arange(0, col, 2).astype(int)
    elif (row <=800 and col <=800):
        rowarr = np.arange(0, row, 4).astype(int)
        colarr = np.arange(0, col, 4).astype(int)
    else:
        rowarr = np.arange(0, row, 10).astype(int)
        colarr = np.arange(0, col, 10).astype(int)

    # rowarr = np.arange(0, row, 2).astype(int)
    # colarr = np.arange(0, col, 2).astype(int)
    # degres=[0, 30,60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    degres=np.arange(0,360,10)
    # if(max_radius-min_radius<=10):
    #     rad = np.arange(min_radius, max_radius,1)
    #     np.append(rad, max_radius)
    # elif (max_radius - min_radius <= 20):
    #     rad = np.arange(min_radius, max_radius, 2)
    #     np.append(rad, max_radius)
    # elif (max_radius - min_radius <= 50):
    #     rad = np.arange(min_radius, max_radius, 1)
    #     np.append(rad, max_radius)
    # else :
    #     rad = np.arange(min_radius, max_radius, 10)
    #     np.append(rad, max_radius)
    if(max_radius-min_radius<=30):
        rad = np.arange(min_radius, max_radius,1)
        np.append(rad, max_radius)
    elif (max_radius - min_radius <= 50):
        rad = np.arange(min_radius, max_radius, 2)
        np.append(rad, max_radius)
    else :
        rad = np.arange(min_radius, max_radius, 3)
        np.append(rad, max_radius)

    arr = np.zeros((row, col, max_radius))


    for i in rowarr:
        # print(i)
        for j in colarr:
            x = edge_img[i][j]
            if x == 1:
                for r in rad:
                    for deg in degres:
                        a = i-r*math.sin(deg*pi/180)
                        b = j-r*math.cos(deg*pi/180)
                        a = int(a)
                        b = int(b)
                        for k in range(a-1,a+2):
                            for l in range(b-1,b+2):
                                if(k > 0 and l > 0 and k < row and l < col):
                                    arr[k][l][r] = arr[k][l][r]+1
    maxes=[]
    for r in rad:
        a=arr[:,:,r]
        maxnum = np.max(a)
        # print( "max for rad ", r ," is ",maxnum)
        num=0
        biggest=0
        bigr=0
        if maxnum not in maxes:
            num+=maxnum
            if(maxnum>biggest):
                biggest=maxnum
                bigr=r
            maxes.append(maxnum)
        # plt.imshow(arr[:,:,r])
        # plt.show()
        middle=math.ceil(num/len(maxes))


    maxes.sort()

    cutoff = maxes[-1]


    listt = []

    # add th emost correct first than add othars
    if (maxnum >= cutoff):
        for i in range(row):
            for j in range(col):
                x = arr[i][j][bigr]
                if (x >= maxnum):
                     listt.append((j, i, r))

    for r in rad:
        a = arr[:, :, r]
        maxnum = np.max(a)
        if(maxnum>middle):
            for i in range(row):
                for j in range(col):
                    x=arr[i][j][r]
                    if (x >= maxnum):
                        # listt.append((j, i, r))
                        add = True
                        # print(listt)
                        for z in range(len(listt)):
                            if (np.abs(listt[z][0] - j) <= min_radius and np.abs(listt[z][1] - i) <= min_radius and np.abs(listt[z][2] - r)<=min_radius):
                                add = False
                                if(np.abs(listt[z][0] - j) <3 and np.abs(listt[z][1] - i) < 3 and np.abs(listt[z][2] - r)<=3):
                                   newj=(listt[z][0] + j)/2.0
                                   newi=(listt[z][1] + i)/2.0
                                   newr=(listt[z][2] + r)/2.0
                                   # print(len(listt))
                                   listt[z] = (newj,newi,newr)
                                   # print(len(listt))
                        if (add):
                            listt.append((j, i, r))
                            # print(x)
    # for i in range(row):
    #     for j in range(col):
    #         for r in rad:
    #             x=arr[i][j][r]
    #             if (x>=cutoff):
    #                 add = True
    #                 # print(listt)
    #                 for z in range(len(listt)):
    #                     if (np.abs(listt[z][0] - j) <= min_radius and np.abs(listt[z][1] - i) <= min_radius):
    #                         add = False
    #                         if(np.abs(listt[z][0] - j) <3 and np.abs(listt[z][1] - i) < 3 and np.abs(listt[z][2] - r)<=3 ):
    #                            newj=(listt[z][0] + j)/2.0
    #                            newi=(listt[z][1] + i)/2.0
    #                            newr=(listt[z][2] + r)/2.0
    #                            print(len(listt))
    #                            listt[z] = (newj,newi,newr)
    #                            print(len(listt))
    #
    #                 if (add):
    #                     listt.append((j, i, r))
    #                     # print(x)


    # cutoff=maxes[-2]
    # for i in range(row):
    #     for j in range(col):
    #         for r in rad:
    #             x=arr[i][j][r]
    #             if (x>=cutoff):
    #                 add=True
    #                 # print(listt)
    #                 for z in range (len(listt)):
    #                     print("listt[z]",listt[z])
    #                     if(np.abs(listt[z][0]-j)<=min_radius and np.abs(listt[z][1]-i)<=min_radius):
    #                         add= False
    #                         if (np.abs(listt[z][0] - j) <= 3 and np.abs(listt[z][1] - i) <=3 and np.abs(listt[z][2] - r)<=3):
    #                             newj = (listt[z][0] + j) / 2.0
    #                             newi = (listt[z][1] + i) / 2.0
    #                             newr = (listt[z][2] + r) / 2.0
    #                             print(len(listt))
    #                             listt[z] = (newj, newi, newr)
    #                             print(len(listt))
    #                 if(add):
    #                     print("adding", j,i,r)
    #                     listt.append((j, i, r))
    #                     # print(x)


    print(listt)
    return listt


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    cv_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)

    shape = in_image.shape
    row = shape[0]
    col = shape[1]
    rowarr = np.arange(0, row, 1).astype(int)
    colarr = np.arange(0, col, 1).astype(int)
    pad= math.floor(k_size/2)
    padded_image = cv2.copyMakeBorder(in_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    image_new = np.zeros(shape)
    # print(shape)

    for x in rowarr:
        print(x)
        for y in colarr:
            pivot_v = in_image[x, y]
            neighbor_hood = padded_image[x :x + k_size ,
                                        y :y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            gaus = cv2.getGaussianKernel(k_size , k_size)
            gaus = gaus.dot(gaus.T)
            combo = gaus * diff_gau
            result = combo * neighbor_hood/ combo.sum()
            ans=result.sum()
            image_new[x][y]=ans

    return cv_image,image_new
