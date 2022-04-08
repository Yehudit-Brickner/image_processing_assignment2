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
    plt.imshow(filterd)
    plt.show()
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
    plt.imshow(new_img)
    plt.show()

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
    # print(img)
    # a=np.array([[1,1,1],[1,1,1],[1,1,1]])
    # a=a/9
    # img=conv2D(img,a)
    # edge_img= edgeDetectionZeroCrossingLOG(img)
    # shape=edge_img.shape
    # row = shape[0]
    # col = shape[1]
    # d3=(shape[0],shape[1],3)
    # copy1=np.zeros(d3)
    # copy2=np.zeros(shape)
    # listt=[]
    #
    # eps=0.1
    #
    # # if(row <200 and col<200):
    # #     rowarr=np.arange(0,row,2)
    # #     colarr = np.arange(0, col, 2)
    # # elif(row <400 and col<400):
    # #     rowarr = np.arange(0, row, 3)
    # #     colarr = np.arange(0, col, 3)
    # # elif (row < 600 and col < 600):
    # #     rowarr = np.arange(0, row, 5)
    # #     colarr = np.arange(0, col, 5)
    # # else:
    # #     rowarr = np.arange(0, row, 10)
    # #     colarr = np.arange(0, col, 10)
    #
    #
    # edge=[]
    # for i in range(row):
    #     for j in range(col):
    #         n = edge_img[i][j]
    #         if n == 1:
    #             edge.append((i,j))
    # print("len of edge=",len(edge))
    # changed_val=[]
    #
    # dif=max_radius-min_radius
    # count=1
    # while(dif>10):
    #     dif=dif/10
    #     count=count+1
    # radarr=np.arange(min_radius,max_radius+1,count)
    #
    #
    # for r in radarr:
    # # for i in rowarr:
    # #     for j in colarr:
    # #         n= edge_img[i][j]
    # #         if n==1:
    #     #for e in edge:
    #     for n in range(0,len(edge),2):
    #         e=edge[n]
    #         i=e[0]
    #         j=e[1]
    #         x_vals = np.arange(i - r, i+ r + 1, 0.25)
    #         y_vals = np.arange(j - r, j + r + 1, 0.25)
    #         for x in x_vals:
    #             for y in y_vals:
    #                 sum = (i - x)**2 +(j - y)**2
    #                 if( sum<r*r+eps and sum>r*r-eps):
    #                     x=x.round()
    #                     y=y.round()
    #                     if(x>0 and y>0 and x<row and y<row):
    #                         copy2[i][j]=copy2[i][j]+0.01
    #                         if(e not in changed_val):
    #                             changed_val.append(e)
    #         #     cv2.circle(copy1,(i,j),r, (255,255,255),1)
    #         #     for k in rowarr:
    #         #         for l in colarr:
    #         #             y=copy1[k][l][0]
    #         #             if y!=0:
    #         #                 copy2[k][l]=copy2[k][l]+0.01
    #         # copy1=copy1*0
    #     print("len of changed_val=", len(changed_val))
    #     count=0
    #     # plt.imshow(copy2)
    #     # plt.show()
    #     max=np.max(copy2)
    #     print(max)
    #     # for i in rowarr:
    #     #     for j in colarr:
    #     for v in changed_val:
    #         i=v[0]
    #         j=v[1]
    #         x=copy2[i][j]
    #
    #     # if x>((min_radius+max_radius)/(2*100)):
    #         if x > (0.35):
    #             count=count+1
    #             listt.append((i,j,r))
    #     print("count for radius ",r ,"is ",count)
    #     copy2=copy2*0
    #     changed_val.clear()
    #
    # return listt
    #
    #



    # # # smooth=np.array([[1,2,1],[2,4,2],[1,2,1]])
    # # # # smooth=smooth/np.sum(smooth)
    # # smooth = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
    # # # # smooth = smooth / np.sum(smooth)
    # # img_smothed = conv2D(img, smooth)
    # # plt.imshow(img_smothed)
    # # plt.show()
    # # lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # # filterd = conv2D(img_smothed, lap_filter)
    # # plt.imshow(filterd)
    # # plt.show()
    # # shape = filterd.shape
    # edge_img= edgeDetectionZeroCrossingLOG(img)
    # shape=edge_img.shape
    # row = shape[0]
    # col = shape[1]
    # rowarr=np.arange(0,row,1).astype(int)
    # # print(rowarr)
    # colarr = np.arange(0, col, 1).astype(int)
    # # print(colarr)
    # pi=math.pi
    # degres=np.arange(0,360,10)
    # degres=[0,90,180,270]
    # print(degres)
    # gap=(int)(max_radius-min_radius)/5
    # rad=np.arange(min_radius,max_radius,2)
    # np.append(rad,max_radius)
    # print(rad)
    # mapp=dict()
    #
    # for i in rowarr:
    #     print(i)
    #     for j in colarr:
    #         x=edge_img[i][j]
    #         if x==1:
    #             for r in rad:
    #                 for deg in degres:
    #     # for i in range(row):
    #     #     for j in range(1):
    #     #         # for r in range():
    #     #         for deg in range(1):
    #     #             r=min_radius
    #                     a= i-r*math.sin(deg*pi/180)
    #                     b=j-r*math.cos(deg*pi/180)
    #                     a=a.round(3)
    #                     b=b.round(3)
    #                     key = (a, b, r)
    #                     # print(key)
    #                     if key in mapp.keys():
    #                         l=mapp.get(key)
    #                         l[1]=l[1]+1
    #                         l[0]=l[0]+deg
    #                         mapp[key]=l
    #                     else:
    #                         mapp[key]=[deg,1]
    #
    #                   # print(mapp[key])
    #
    # print("mapp size=", len(mapp))
    # map2=dict()
    # for x in mapp:
    #     key=(x[0],x[1])
    #     val=mapp.get(x)
    #     if key in map2.keys():
    #         l=map2.get(key)
    #         l[0]=l[0]+val[0]
    #         l[1]=l[1]+val[1]
    #         l[2]=l[2]+x[2]
    #         l[3]=l[3]+1
    #     else:
    #         map2[key]=[val[0],val[1],x[2],1]
    # print("map2 size",len(map2))
    #
    # count=0
    # listt=[]
    # for x in map2:
    #     # print("x=",x)
    #     val=map2.get(x)
    #
    #     if val[1]>38:
    #         print("key=",x)
    #         print("val=", val)
    #         count=count+1
    #         deg=round(val[0]/val[1])%360
    #         r=val[2]/val[3]
    #         i=x[0]+r*math.sin(deg*pi/180)
    #         i=round(i)
    #         # i=i%row
    #         j=x[1]+r*math.cos(deg*pi/180)
    #         j=round(j)
    #         # j=j%col
    #         # i=val[0]
    #         # j=val[1]
    #         ci=(i,j,r)
    #         listt.append(ci)
    # print(count)
    # print(listt)
    #
    # return listt


    edge_img= edgeDetectionZeroCrossingLOG(img)
    shape=edge_img.shape
    row = shape[0]
    col = shape[1]
    rowarr=np.arange(0,row,1).astype(int)

    colarr = np.arange(0, col, 1).astype(int)

    pi=math.pi

    degres=[0,30,60,90,120,150,180,210,240,270,300,330]

    rad=np.arange(min_radius,max_radius)
    np.append(rad,max_radius)
    print(rad)

    arr=np.zeros((row,col,max_radius))

    for i in rowarr:
        print(i)
        for j in colarr:
            x=edge_img[i][j]
            if x==1:
                for r in rad:
                    for deg in degres:
                        a= i-r*math.sin(deg*pi/180)
                        b=j-r*math.cos(deg*pi/180)
                        a=round(a)
                        a=int(a)
                        b=round(b)
                        b=int(b)
                        # print(a,b,r)
                        if(a>0 and b>0 and a<row and b<col):
                            arr[a][b][r]=arr[a][b][r]+1

    listt=[]
    for i in rowarr:
        print(i)
        for j in colarr:
                for r in rad:
                    x=arr[i][j][r]
                    if (x>8):
                        listt.append((i,j,r))
                        listt.append((i+int(row/10),j+int(col/10),r))
                        listt.append((i + 2*int(row / 10), j + 2*int(col / 10), r))


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

    return
