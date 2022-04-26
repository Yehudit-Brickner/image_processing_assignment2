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
    # check if the in_signal is bigger than the kernel, if not switch them
    if (len(k_size) > len(in_signal)):
        temp = k_size.copy()
        k_size = in_signal.copy()
        in_signal = temp.copy()

    # the amount of zeros needed to pad each end of the in_signal
    shift = (len(k_size) - 1)

    # creating the new array that will be returned
    new = np.zeros(len(in_signal) + shift)

    # flipping the kernel
    flip = np.flip(k_size)

    # creating a copy of the in_signal and pad it with zeros
    padded = in_signal.copy()
    for k in range(shift):
        padded = np.insert(0, 0, padded)
        padded = np.append(0, padded)

    # doing the multiplication for each value and putting into its spot in the new array
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

    # finding half of the kernels row and column to know how much to pad the image
    r_skip = math.floor(ker_row / 2)
    c_skip = math.floor(ker_col / 2)

    # creating a padded image
    padded_image = cv2.copyMakeBorder(in_image, r_skip, r_skip, c_skip, c_skip, cv2.BORDER_REPLICATE, None, value=0)

    # doing the multiplication to find the new value for each pixel
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

    # create a 2d array by multiplying by its transpose
    arr_new=arr*arr.T

    # divide by its sum
    arr_new=arr_new/np.sum(arr_new)
    in_image=in_image*255
    # create a new image using conv2d
    img_new=conv2D(in_image, arr_new)
    img_new=img_new/255
    return img_new


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    blur = cv2.GaussianBlur(in_image, (k_size, k_size),-1)
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
    # get an array of the 7th row of pascals triangle
    # and crete a kernel from the array
    smooth=pascal_triangel(7)
    smooth=smooth*smooth.T

    # smoothing the image with the kernel
    img_smothed=conv2D(img,smooth)

    # laplacian filter
    lap_filter=np.array([[0,1,0],[1,-4,1],[0,1,0]])

    # filtering the image with the Laplacian kernel to get the 2nd derivative of the image
    filterd = conv2D(img_smothed,lap_filter)

    shape = filterd.shape
    row = shape[0]
    col = shape[1]
    new_img = np.zeros(shape)


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

    return new_img


def edgeDetectionZeroCrossingLOG5(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    # same as the function above but smoothing the image first with a 5x5 kernel instead of a 7x7 kernel.
    # this works better for hough circles

    smooth = pascal_triangel(5)
    smooth = smooth*smooth.T
    img_smothed = conv2D(img,smooth)
    lap_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    filterd = conv2D(img_smothed,lap_filter)
    shape = filterd.shape
    row = shape[0]
    col = shape[1]
    new_img = np.zeros(shape)


    # we will not include the outer edges of the image so that we can find the other edges easier
    # we will look for places thar are {++0--} or {--0++} or {-++} or {+--} in the horizontal or vertical axis
    # where the 0 is the spot i am looking at or the lone -/+ is the spot i am looking at
    # if the spot is a edge it will come up true for one of the options
    for i in range (2,row-2):
        for j in range(2,col-2):
            center = filterd[i][j]
            up = filterd[i-1][j]
            up2 =filterd[i-2][j]
            down = filterd[i+1][j]
            down2 = filterd[i+2][j]
            left = filterd[i][j-1]
            left2 = filterd[i][j-2]
            right = filterd[i][j+1]
            right2 = filterd[i][j+2]
            if(center == 0):
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

    """
    the threshold for finding the circle depend om the picture and the values in the hough transform
    """



    # makeing a 3x3 kernel to blur the image
    ker=np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    img=conv2D(img,ker)
    # finding the edges
    edge_img = edgeDetectionZeroCrossingLOG5(img)

    shape = edge_img.shape
    row = shape[0]
    col = shape[1]
    pi = math.pi

    # to find hough circles you need to take all pixels that are edges and find where they map to in the hough space for all radius and all degres,
    # but that is a lot to calcultae espiecially if it is a big image and there are a lot of edges and a big span of radius to cheeck
    # so i decided that depending on the size of the image i will only check part of the column and rows.
    # and that i will check part of the radius depending or how many there are.
    # and that the degrees will be in increments of 12 always.

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

    # creating an 3d array to be the hough space
    arr = np.zeros((row, col, max_radius))


    # we will go through the image and if the pixel is an edge, we will transform to the hough space.
    # to transform to the hough space we used the algo in rows 412-413.
    # we will change the a and b to ints so that we can mark the spot in the hough space array
    # we will mark a 3x3 square around that pixel if its in the array we will make the cenre bigger by 1.2 and theotside bigger by 1
    # the algorthm for the transform was taken from wikipedia
    # https://en.wikipedia.org/wiki/Circle_Hough_Transform
    for i in rowarr:
        for j in colarr:
            x = edge_img[i][j]
            if x != 0:
                for r in rad:
                    for deg in degres:
                        a = i-r*math.sin(deg*pi/180)
                        b = j-r*math.cos(deg*pi/180)
                        a = int(a)
                        b = int(b)
                        for k in range(a-1,a+2):
                            for l in range(b-1,b+2):
                                if(k >= 0 and l >= 0 and k < row and l < col):
                                    if(k==a and l==b):
                                        arr[k][l][r] = arr[k][l][r] + 1.2
                                    else:
                                        arr[k][l][r] = arr[k][l][r]+1


    # we will go through the 3d array and find for each raidus that we are searching
    # what the max value is and add it a list maxes if not in it
    # we will sort the array and find sum of the array.
    # depending on the size of list we will crete a variable called middle
    # middle will be some sort of average of part of the list
    maxes=[]
    for r in rad:
        a=arr[:,:,r]
        maxnum = np.max(a)
        if maxnum not in maxes:
            maxes.append(maxnum)
    maxes.sort()
    num=sum(maxes)
    # print(maxes)
    if(len(maxes)>4):
        middle=math.ceil((num-maxes[-1]-maxes[-2]-maxes[-3])/(len(maxes)-3))
    else:
        if(len(maxes)>=2):
            middle=math.ceil(num-maxes[0]/len(maxes)-1)
        else:
            middle=maxes[0]
    # print("middle=",middle)

    # we will create a list that will return at the end
    # the list will have tuples with these values- middle of the circle and radius, and correctness
    # we will go through the 3d array and check the layers that the max value is the same as cutoff
    # we will go through those layers and if the pixels value is bigger or equal to cutoff we will add it to the list
    # when we wdd it to the list we will switch the i and j cordinates so that the middle of the circle is in the correct position in the picture
    thresholds = []
    listt = []
    cutoff = maxes[-1] # biggest value in maxes
    thresholds.append(cutoff)
    for r in rad:
        a = arr[:, :, r]
        maxnum = np.max(a)
        if(maxnum==cutoff):
            for i in range(row):
                for j in range(col):
                    x = arr[i][j][r]
                    if (x >= cutoff):
                         listt.append((j, i, r,x))

    # we will repeat the same thing for all layers that maxnum is bigger or equal to middle
    # if the spot is bigger or equal to the average of maxnum and middle
    # but before dding the circle to the list we will check if it is really similar to a circle in the list
    # if it is similar if the i j and r are all within min_radius/2 from a circle that exsits
    # if its in that area we will check if the difference is up to 3 away
    # if so we will update that entry in the list by averaging the values

    c = min_radius / 2
    for r in rad:
        a = arr[:, :, r]
        maxnum = np.max(a)
        if(maxnum>=middle):
            threshold=(maxnum+middle)/2
            thresholds.append(threshold)
            for i in range(row):
                for j in range(col):
                    x=arr[i][j][r]
                    if (x >= threshold):
                        add = True
                        for z in range(len(listt)):
                            if (np.abs(listt[z][0] - j) <= c and np.abs(listt[z][1] - i) <= c and np.abs(listt[z][2] - r)<= c):
                                add = False
                                if(np.abs(listt[z][0] - j) <3 and np.abs(listt[z][1] - i) < 3 and np.abs(listt[z][2] - r)<=3):
                                   newj = (listt[z][0] + j)/2.0
                                   newi = (listt[z][1] + i)/2.0
                                   newr = (listt[z][2] + r)/2.0
                                   newx = (listt[z][3] + x)/2.0
                                   listt[z] = (newj,newi,newr, newx)
                        if (add):
                            listt.append((j, i, r,x))


    # going through the list and removing smaller circle from bigger circle if the bigger circle is more accurate
    remove_list=[]
    for l in range(len(listt)):
        for k in range(len(listt)):
            if(l!=k):
                inside=in_circle(listt[l][0],listt[l][1],listt[l][2],listt[l][3],listt[k][0],listt[k][1],listt[k][2],listt[k][3])
                if inside==1:
                    if l not in remove_list:
                        remove_list.append(l)
    remove_list.sort(reverse=True)
    # print(remove_list)
    for x in remove_list:
        del listt[x]

    # print(listt)
    print("best thresholds are:", thresholds)
    return listt


def in_circle( x1, y1, r1,ac1, x2, y2, r2, ac2 ):
    # function to check if a circle is inside another circle
    if(r2>r1 and ac2>ac1):
        dist=math.sqrt((x1-x2)**2 +(y1-y2)**2)
        if dist<r2:
            return 1
    return 0



def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """



    # cv bilateral
    cv_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)

    shape = in_image.shape
    row = shape[0]
    col = shape[1]
    rowarr = np.arange(0, row, 1).astype(int)
    colarr = np.arange(0, col, 1).astype(int)
    pad= math.floor(k_size/2)
    # creating a padded image
    padded_image = cv2.copyMakeBorder(in_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    image_new = np.zeros(shape)

    # creating a gaus kernel
    gaus = cv2.getGaussianKernel(k_size, sigma_space)
    gaus = gaus.dot(gaus.T)

    # creating all the kernels needed and combining them together and multiplying by the neighborhood
    # putting into the new image the sum of the all that
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
