import cv2
import numpy as np
import matplotlib.pyplot as plt


def task2_image_conversions(path: str):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # пороговая бинаризация
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # адаптивная бинаризация
    adaptive = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    cv2.imshow("Original", img)
    cv2.imshow("Gray", gray)
    cv2.imshow("Binary (128)", binary)
    cv2.imshow("Adaptive Binary", adaptive)

    cv2.imwrite("2_gray.png", gray)
    cv2.imwrite("2_binary.png", binary)
    cv2.imwrite("2_adaptive.png", adaptive)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task3_hist_equalize(path: str):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    eq = cv2.equalizeHist(img)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original histogram")
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.title("Equalized histogram")
    plt.hist(eq.ravel(), bins=256, range=(0, 256), color='black')
    plt.xlabel("Intensity")

    plt.tight_layout()
    plt.savefig("3_histograms.png")
    plt.show()

    cv2.imshow("Original Dark", img)
    cv2.imshow("Equalized", eq)
    cv2.imwrite("3_equalized.png", eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task4_noise_filtering(path: str):
    img = cv2.imread(path)

    blur       = cv2.blur(img, ksize=(5, 5))
    gaussian   = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
    median     = cv2.medianBlur(img, ksize=5)

    cv2.imshow("Original Noisy", img)
    cv2.imshow("Blur", blur)
    cv2.imshow("Gaussian Blur", gaussian)
    cv2.imshow("Median Blur", median)

    cv2.imwrite("4_blur.png", blur)
    cv2.imwrite("4_gaussian.png", gaussian)
    cv2.imwrite("4_median.png", median)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task5_morphology(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)

    eroded  = cv2.erode(binary,  kernel, iterations=1)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    cv2.imshow("Original Objects", img)
    cv2.imshow("Binary", binary)
    cv2.imshow("Eroded", eroded)
    cv2.imshow("Dilated", dilated)

    cv2.imwrite("5_binary.png", binary)
    cv2.imwrite("5_eroded.png", eroded)
    cv2.imwrite("5_dilated.png", dilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 2. Изменение формата изображения
    # task2_image_conversions("image.png")
    #
    # 3. Гистограмма и выравнивание освещённости
    # task3_hist_equalize("image2.jpg")
    #
    # # 4. Фильтрация шумного изображения
    #task4_noise_filtering("image3_1.png")
    #
    # # 5. Морфологические операции на бинарном изображении
    task5_morphology("image4.jpg")

