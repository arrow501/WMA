import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk

# instalki
# pip install opencv-python
# pip install matplotlib


# Funkcja zmieniająca rozmiar obrazu do rozmiaru ekranu
def resize_image_to_screen_size():
    global image
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    h, w = image.shape[:2]
    if w > h:
        image = cv2.resize(image, (screen_width, int(screen_width*h/w)), interpolation= cv2.INTER_LINEAR)
    else:
        image = cv2.resize(image, (int(screen_height*w/h), screen_height), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

# Funkcja zmniejszająca rozmiar obrazu o 10%
def resize():
    global image
    h, w = image.shape[:2]
    h = h + int(h*(-0.1))
    w = w + int(w*(-0.1))
    image = cv2.resize(image, (w, h), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

# Funkcja stosująca filtr Canny do obrazu
def image_canny():
    global image
    high_color = cv2.getTrackbarPos('high','obrazek')
    b=cv2.blur(image, (high_color,high_color))
    # cv2.imshow('obrazek', b)

    cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))

# Funkcja zmieniająca kolor obrazu na podstawie wartości z trackbara
def change_color():
    global image

    colorGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('obrazek1', color)
    
    # cv2.imshow('obrazek0', color[:,:,0])
    # cv2.imshow('obrazek1', color[:,:,1])
    # cv2.imshow('obrazek2', color[:,:,2])

    # cv2.imshow('obrazekSzary', colorGray)
     
    # cv2.imshow('obrazek', color[:,:,1])

    high_color = cv2.getTrackbarPos('high','obrazek')
    lower = np.array([0,0,0]) 
    upper = np.array([high_color,255,255]) 
    mask = cv2.inRange(color, lower, upper) 
    cv2.imshow('mask', mask)
    res = cv2.bitwise_and(image, image, mask=mask) 
    cv2.imshow('bitwise', res)
    res = cv2.medianBlur(res, ksize=5) 
    cv2.imshow('blur', res)

# Funkcja wywoływana przy zmianie wartości trackbara
def change_h(x):
    global fun
    fun()

image = None
fun = None

# Główna funkcja programu
def main():
    global image
    global fun
    global files
    
    # Pobranie listy plików z katalogu
    files = os.listdir(r'C:\Users\macmac\Downloads\pliki')

    # Wczytanie pierwszego obrazu z listy
    image = cv2.imread(r'C:\Users\macmac\Downloads\pliki\{0}'.format(files[0]))
    
    # Zmiana rozmiaru obrazu do rozmiaru ekranu
    resize_image_to_screen_size()
    cv2.createTrackbar('high','obrazek',0,255,change_h)
    fun = None
    while True:
        key = cv2.waitKey()
        if key == ord('-'):
            fun = None
            resize()
        if key == ord('c'):
            fun = image_canny
            fun()
        if key == ord('k'):
            fun = change_color
            fun()
        if key <= ord('9') and key >= ord('0'):
            fun = None
            image = cv2.imread(r'C:\Users\macmac\Downloads\pliki\{}'.format(files[key-ord('0')]))
            resize_image_to_screen_size()
        elif key == 27:
            cv2.destroyAllWindows()
            break

if __name__=='__main__': 
    main()