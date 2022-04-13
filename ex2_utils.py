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
    # check if the in_sugnal is bigger than the kernel, if not switch them
    if (len(k_size) > len(in_signal)):
        temp = k_size.copy()
        k_size = in_signal.copy()
        in_signal = temp.copy()

    # the amount of zeros needed to pad each end of the in_signal
    shift = (len(k_size) - 1)

    # creating the new array that will be returned
    new = np.zeros(len(in_signal) + shift)

    # fliiping the kernel
    flip = np.flip(k_size)

    # creating a copy of the in_signal and paddig with zeros
    padded = in_signal.copy()
    for k in range(shift):
        padded = np.insert(0, 0, padded)
        padded = np.append(0, padded)

    # doing the mulitiplacation for each value and putting into its spot in the new array
    for i in range(len(padded) - shift):
        num = 0
        for k in range(len(k_size)):
            num += flip[k] * padded[i + k]
        new[i] = num
    return new


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    # finding the size of the row and col for the image and the kernel
    shape_ker = kernel.shape
    ker_row = shape_ker[0]
    ker_col = shape_ker[1]

    shape_img = in_image.shape
    img_row = shape_img[0]
    img_col = shape_img[1]

    # creating a new array the size of the image
    new_img_blurred = np.zeros(shape_img)

    # flipping the kernel
    kernel = np.flip(kernel)

    # finding half of the kerels row and column to know how much to pad the image
    r_skip = math.floor(ker_row / 2)
    c_skip = math.floor(ker_col / 2)

    # creating a padded image
    padded_image = cv2.copyMakeBorder(in_image, r_skip, r_skip, c_skip, c_skip, cv2.BORDER_REPLICATE, None, value=0)

    # doing the multiplacation to find the new value for each pixel
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

    # this is the same as conv2d but the padded image is reflected and not replicated

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
    # kernel for derivative using x
    ker = np.array([[1, 0, -1]])

    # create the transpose for the y derivative
    kerT = np.transpose(ker)

    # create new images- using conv2d_reflect because cv2 uses reflect when calc the derivative
    img_x = conv2D_REFLECT(in_image, ker)
    img_y = conv2D_REFLECT(in_image, kerT)

    # calculating the magnitude and the direction
    MagG = np.sqrt((img_x * img_x ) + (img_y * img_y))
    dirG = np.arctan2(img_y, img_x).astype(np.float64)
    return dirG, MagG


def pascal_triangel( row: int)->np.ndarray:

    #creating an array
    arr=np.zeros((row,row))

    #putting in the first to rows of pascals triangle
    arr[0][0]=1
    arr[1][0]=1
    arr[1][1]=1

    # calculating the rest of the triangle
    for i in range(2,row):
        for j in range (0,row):
            if j==0: # the first column will always be a 1
                arr[i][j]=1
            else:
                arr[i][j] = arr[i-1][j]+arr[i-1][j-1]

    # get the last row of the array
    arr_new=arr[row-1:row,:]
    return arr_new

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # get the k_size row of the pascal triangle
    arr=pascal_triangel(k_size)

    # create a 2d array by multipling by its transpose
    arr_new=arr*arr.T

    # divide by its sum
    arr_new=arr_new/np.sum(arr_new)

    # create a new image using conv2d_reflect
    img_new=conv2D_REFLECT(in_image, arr_new)

    return img_new


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    blur = cv2.GaussianBlur(in_image, (k_size, k_size), 0)

    return blur


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
    # smooth=smooth/np.sum(smooth)
    # smooth=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    smooth = np.array([[0,0,1,2,1,0,0], [0,3,13,22,13,3,0], [1,13,59,97,59,13,1], [2,22,97,159,97,22,2],[1,13,59,97,59,13,1],[0,3,13,22,13,3,0],[0,0,1,2,1,0,0]])
    smooth=pascal_triangel(7)
    smooth=smooth*smooth.T
    # print(smooth)
    # smooth = smooth / np.sum(smooth)
    img_smothed=conv2D(img,smooth)
    # img_smothed=cv2.filter2D(img,-1, smooth, borderType=cv2.BORDER_REPLICATE)
    # img_smothed = cv2.GaussianBlur(img, (7, 7), 0)
    # plt.imshow(img_smothed)
    # plt.show()
    lap_filter=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    filterd=conv2D(img_smothed,lap_filter)
    # plt.imshow(filterd)
    # plt.show()
    # filterd=cv2.filter2D(img_smothed,-1,lap_filter, borderType=cv2.BORDER_REPLICATE)
    # plt.imshow(filterd)
    # plt.show()
    shape=filterd.shape
    row=shape[0]
    col=shape[1]
    new_img=np.zeros(shape)
    # we will not include the outer edges of the image so that we can find the other edges easier
    # for i in range (1,row-1):
    #     for j in range(1,col-1):
    #         center =filterd[i][j]
    #         up= filterd[i-1][j]
    #         down =filterd[i+1][j]
    #         left= filterd[i][j-1]
    #         right=filterd[i][j+1]
    #         if(center==0):
    #             if(right*left<0):
    #                 new_img[i][j]=1
    #             if(up*down<0):
    #                 new_img[i][j] = 1
    #         elif (center*up<0):
    #             new_img[i][j] = 1
    #             new_img[i-1][j] = 1
    #         elif (center * down < 0):
    #             new_img[i][j] = 1
    #             new_img[i+1][j] = 1
    #         elif (center*left<0):
    #             new_img[i][j] = 1
    #             new_img[i][j-1] = 1
    #         elif (center*right<0):
    #             new_img[i][j] = 1
    #             new_img[i][j+1] = 1

    # we will not include the outer edges of the image so that we can find the other edges easier
    # we will look for places thar are {++0--} or {--0++} or {-++} or {+--} in the horizontal or vertical axis
    # where the 0 is the spot i am looking at or the lone -/+ is the spot i am looking at
    # if the spot is a edge it will come up true for one of the options
    for i in range (2,row-2):
        for j in range(2,col-2):
            center =filterd[i][j]
            up= filterd[i-1][j]
            up2=filterd[i-2][j]
            down =filterd[i+1][j]
            down2 = filterd[i+2][j]
            left= filterd[i][j-1]
            left2 = filterd[i][j-2]
            right=filterd[i][j+1]
            right2 = filterd[i][j+2]
            if(center==0):
                if(right*left<0 and right2*left2<0 and right2*right>0 and left2*left>0):
                    new_img[i][j]=1
                if(up*down<0 and up2*down<0 and up*up2>0 and down2*down>0):
                    new_img[i][j] = 1
            elif (center*up<0 and center*up2<0 and up2*up>0):
                new_img[i][j] = 1
                new_img[i-1][j] = 1
            elif (center * down < 0 and center*down2<0 and down2*down>0):
                new_img[i][j] = 1
                new_img[i+1][j] = 1
            elif (center*left<0 and center*left2<0 and left2*left>0):
                new_img[i][j] = 1
                new_img[i][j-1] = 1
            elif (center*right<0 and center*right2<0 and right2*right>0):
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


    if (row <=300 and col <=300):
        rowarr = np.arange(0, row, 1).astype(int)
        colarr = np.arange(0, col, 1).astype(int)
    elif (row <=450and col <=450):
        rowarr = np.arange(0, row, 2).astype(int)
        colarr = np.arange(0, col, 2).astype(int)
    elif (row <= 600 and col <=600):
        rowarr = np.arange(0, row, 3).astype(int)
        colarr = np.arange(0, col, 3).astype(int)
    elif (row <=800 and col <=800):
        rowarr = np.arange(0, row, 4).astype(int)
        colarr = np.arange(0, col, 4).astype(int)
    else:
        rowarr = np.arange(0, row, 10).astype(int)
        colarr = np.arange(0, col, 10).astype(int)

    if(max_radius-min_radius<=30):
        rad = np.arange(min_radius, max_radius,1)
        np.append(rad, max_radius)
    elif (max_radius - min_radius <= 50):
        rad = np.arange(min_radius, max_radius, 2)
        np.append(rad, max_radius)
    else :
        rad = np.arange(min_radius, max_radius, 3)
        np.append(rad, max_radius)

    degres = np.arange(0, 360, 12)
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
                                if(k >= 0 and l >= 0 and k < row and l < col):
                                    arr[k][l][r] = arr[k][l][r]+1
    maxes=[]
    # num = 0
    for r in rad:
        a=arr[:,:,r]
        maxnum = np.max(a)
        if maxnum not in maxes:
            # n\um+=maxnum
            maxes.append(maxnum)
        # plt.imshow(arr[:,:,r])
        # plt.show()
    maxes.sort()
    num=sum(maxes)
    print(maxes)
    cutoff = maxes[-1]
    # print("num=",num)
    if(len(maxes)>4):
        middle=math.ceil((num-cutoff-maxes[-2]-maxes[-3])/(len(maxes)-3))
    else:
        if(len(maxes)>=2):
            middle=math.ceil(num-maxes[0]/len(maxes)-1)
        else:
            middle=maxes[0]
    print("middle=",middle)

    listt = []

    for r in rad:
        a = arr[:, :, r]
        maxnum = np.max(a)
        if(maxnum==cutoff):
            for i in range(row):
                for j in range(col):
                    x = arr[i][j][r]
                    if (x >= cutoff):
                         listt.append((j, i, r))

    for r in rad:
        a = arr[:, :, r]
        maxnum = np.max(a)
        if(maxnum>=middle):
            for i in range(row):
                for j in range(col):
                    x=arr[i][j][r]
                    if (x >= maxnum):
                        add = True
                        for z in range(len(listt)):
                            if (np.abs(listt[z][0] - j) <= min_radius and np.abs(listt[z][1] - i) <= min_radius and np.abs(listt[z][2] - r)<=min_radius):
                                add = False
                                if(np.abs(listt[z][0] - j) <3 and np.abs(listt[z][1] - i) < 3 and np.abs(listt[z][2] - r)<=3):
                                   newj=(listt[z][0] + j)/2.0
                                   newi=(listt[z][1] + i)/2.0
                                   newr=(listt[z][2] + r)/2.0
                                   listt[z] = (newj,newi,newr)
                        if (add):
                            listt.append((j, i, r))
    print(listt)
    return listt


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (np.ndarray, np.ndarray):
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


    gaus = cv2.getGaussianKernel(k_size, k_size )
    gaus = gaus.dot(gaus.T)

    for x in rowarr:
        for y in colarr:
            pivot_v = in_image[x, y]
            neighbor_hood = padded_image[x :x + k_size ,
                                        y :y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = gaus * diff_gau
            result = combo * neighbor_hood/ combo.sum()
            ans=result.sum()
            image_new[x][y]=ans

    return cv_image,image_new
