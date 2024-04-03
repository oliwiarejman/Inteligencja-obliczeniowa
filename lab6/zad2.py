import cv2
import matplotlib.pyplot as plt
import numpy as np


def konwersja_srednia(img):
    return (img[:,:,2]+img[:,:,1]+img[:,:,0])/3

def konwersja_wzor(img):
    return 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]

def pokaz_obrazy(original, gray_avg, gray_formula):
    plt.figure(figsize=(12, 6))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Obraz oryginalny')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(gray_avg, cmap='gray')
    plt.title('Skala szarości (średnia)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(gray_formula, cmap='gray')
    plt.title('Skala szarości (wzór)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

img1 = cv2.imread('obraz1.jpg')
img2 = cv2.imread('obraz2.jpg')
img3 = cv2.imread('obraz3.jpg')

gray_avg_img1 = konwersja_srednia(img1)
gray_avg_img2 = konwersja_srednia(img2)
gray_avg_img3 = konwersja_srednia(img3)

gray_formula_img1 = konwersja_wzor(img1)
gray_formula_img2 = konwersja_wzor(img2)
gray_formula_img3 = konwersja_wzor(img3)

pokaz_obrazy(img1, gray_avg_img1, gray_formula_img1)
pokaz_obrazy(img2, gray_avg_img2, gray_formula_img2)
pokaz_obrazy(img3, gray_avg_img3, gray_formula_img3)
