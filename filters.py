import matplotlib.pyplot as mp_plt
import matplotlib.image as mp_img
import numpy
from skimage import color
import sys


def generate_box_1d(k):
    box_filter = numpy.ones(k, dtype=numpy.int)
    box_factor = 1 / k
    return box_filter, box_factor


def generate_box_2d(k):
    box_filter = numpy.array([numpy.ones(k, dtype=numpy.int).tolist() for _ in range(k)])
    box_factor = 1 / k ** 2
    return box_filter, box_factor


def generate_gauss_1d(N=3, sigma=1):
    n = (N - 1.) / 2.
    x = numpy.ogrid[-n:n + 1]
    h = numpy.exp(-(x ** 2) / (2. * sigma ** 2)) / (2 * numpy.pi * sigma ** 2)
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h, 1


def generate_gauss_2d(shape=(3, 3), sigma=1):
    m, n = [(dim - 1.) / 2. for dim in shape]
    y, x = numpy.ogrid[-m:m + 1, -n:n + 1]
    h = numpy.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2)) / (2 * numpy.pi * sigma ** 2)
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h, 1


def convolution_2d(image, kernel, factor):
    output = numpy.zeros_like(image)
    image_padded = numpy.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))

    image_padded[int((kernel.shape[0] - 1) / 2):-int(kernel.shape[0] / 2),
    int((kernel.shape[1] - 1) / 2):-int(kernel.shape[1] / 2)] = image

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = factor * (image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel).sum()
    return output


def find_edges(image, kernel_x, kernel_y, factor):
    """
        Find edges using sobel operator.
    """
    output = numpy.zeros_like(image)
    image_paddedx = numpy.zeros((image.shape[0] + kernel_x.shape[0] - 1, image.shape[1] + kernel_x.shape[1] - 1))
    image_paddedy = numpy.zeros((image.shape[0] + kernel_y.shape[0] - 1, image.shape[1] + kernel_y.shape[1] - 1))

    image_paddedx[int((kernel_x.shape[0] - 1) / 2):-int(kernel_x.shape[0] / 2),
    int((kernel_x.shape[1] - 1) / 2):-int(kernel_x.shape[1] / 2)] = image
    image_paddedy[int((kernel_y.shape[0] - 1) / 2):-int(kernel_y.shape[0] / 2),
    int((kernel_y.shape[1] - 1) / 2):-int(kernel_y.shape[1] / 2)] = image

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            g_x = factor * (image_paddedx[y:y + kernel_x.shape[0], x:x + kernel_x.shape[1]] * kernel_x).sum()
            g_y = factor * (image_paddedy[y:y + kernel_y.shape[0], x:x + kernel_y.shape[1]] * kernel_y).sum()
            g = numpy.sqrt(g_x ** 2 + g_y ** 2)
            output[y, x] = g * factor
    return output


def main():
    img = mp_img.imread("ss2.jpg")
    img = color.rgb2gray(img)

    mp_plt.figure(1)
    mp_plt.title('ORIGINAL')
    mp_plt.imshow(img, cmap='gray')

    if str(sys.argv[1]) == "1":
        box_filter, box_factor = generate_box_2d(3)
        res = convolution_2d(img, box_filter, box_factor)

        mp_plt.figure(2)
        mp_plt.title('BOX_3x3')
        mp_plt.imshow(res, cmap='gray')

        box_filter, box_factor = generate_box_2d(5)
        res = convolution_2d(img, box_filter, box_factor)

        mp_plt.figure(3)
        mp_plt.title('BOX_5x5')
        mp_plt.imshow(res, cmap='gray')

        box_filter, box_factor = generate_box_2d(7)
        res = convolution_2d(img, box_filter, box_factor)

        mp_plt.figure(4)
        mp_plt.title('BOX_7x7')
        mp_plt.imshow(res, cmap='gray')

        gauss_filter, gauss_factor = generate_gauss_2d((3, 3))
        res = convolution_2d(img, gauss_filter, gauss_factor)

        mp_plt.figure(5)
        mp_plt.title('GAUSS_3x3')
        mp_plt.imshow(res, cmap='gray')

        gauss_filter, gauss_factor = generate_gauss_2d((5, 5))
        res = convolution_2d(img, gauss_filter, gauss_factor)

        mp_plt.figure(6)
        mp_plt.title('GAUSS_5x5')
        mp_plt.imshow(res, cmap='gray')

        gauss_filter, gauss_factor = generate_gauss_2d((7, 7))
        res = convolution_2d(img, gauss_filter, gauss_factor)

        mp_plt.figure(7)
        mp_plt.title('GAUSS_7x7')
        mp_plt.imshow(res, cmap='gray')
    elif str(sys.argv[1]) == "2":
        sobel_x = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_factor = 1

        res = find_edges(img, sobel_x, sobel_y, edge_factor)

        mp_plt.figure(2)
        mp_plt.title('EDGES')
        mp_plt.imshow(res, cmap='gray')
    elif str(sys.argv[1]) == "3":
        cropped_img = mp_img.imread("ss1_crop1.jpg")
        cropped_img = color.rgb2gray(cropped_img)
        factor = 1

        res = convolution_2d(img, cropped_img, factor)

        mp_plt.figure(2)
        mp_plt.title('BOUNDING_BOX1')
        mp_plt.imshow(res, cmap='gray')

        cropped_img = mp_img.imread("ss1_crop2.jpg")
        cropped_img = color.rgb2gray(cropped_img)
        res = convolution_2d(img, cropped_img, factor)

        mp_plt.figure(3)
        mp_plt.title('BOUNDING_BOX2')
        mp_plt.imshow(res, cmap='gray')

    mp_plt.show()


if __name__ == '__main__':
    main()
