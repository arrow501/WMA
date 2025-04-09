import cv2
import numpy as np
import os

# pip install opencv-python
image = None
fun = None
files = None
path_pliki = r'CW05\pliki'

def uploud(i):
    global files, image
    image = cv2.imread(r'{}\{}'.format(path_pliki, files[i-ord('0')]))
    norm_size()

# key -
def resize():
    global image
    h, w = image.shape[:2]
    h = h + int(h*(-0.1))
    w = w + int(w*(-0.1))
    image = cv2.resize(image, (w, h), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)


def norm_size():
    global image
    h, w = image.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800/h)) * (-1)
            w = w + int(w*(s))
            h = h + int(h*(s))
            image = cv2.resize(image, (w, h), interpolation= cv2.INTER_LINEAR)
    else:
        if w > 800:
            s = (1 - (800/w)) * (-1)
            w = w + int(w*(s))
            h = h + int(h*(s))
            image = cv2.resize(image, (w, h), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

# key e
def hsv_range():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    # Convert the HSV colorspace
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue color
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    cv2.imshow('obrazek', mask)

# key r
def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)

# key t
def hsv_median():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)

# key f
def morphology():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)

# key g
def morphology2():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((ksize, ksize), np.uint8)
    # mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)

# key h
def marker():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    print(contours)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)

# key p
def connect_mask():
    # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    # Konwersja obrazu na przestrzeń kolorów HSV
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Utworzenie maski dla pierwszego zakresu kolorów
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Nałożenie maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('mask 1', res)

    # Utworzenie maski dla drugiego zakresu kolorów
    lower = np.array([0, 100, 100])
    upper = np.array([ksize, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower, upper)

    # Nałożenie drugiej maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=mask2)
    cv2.imshow('mask 2', res)

    # Połączenie dwóch masek za pomocą operacji bitowej OR
    b_mask = cv2.bitwise_or(mask, mask2)

    # Nałożenie połączonej maski na obraz i wyświetlenie wyniku
    res = cv2.bitwise_and(image, image, mask=b_mask)
    cv2.imshow('obrazek', res)

# key j
def find_circle():
    # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    # Utwórz kopię obrazu, aby nie modyfikować oryginału
    c_img = image.copy()

    # Konwersja obrazu na skalę szarości
    gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)

    # Zastosowanie rozmycia na obrazie w skali szarości
    bimg = cv2.blur(gimg, (ksize, ksize))

    # Wykrywanie okręgów za pomocą transformacji Hougha
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, high_color, low_color)
    print(circles)  # Wyświetlenie wykrytych okręgów (surowe dane)

    # Zaokrąglenie współrzędnych wykrytych okręgów do liczb całkowitych
    circles = np.uint16(np.around(circles))
    print(circles)  # Wyświetlenie zaokrąglonych współrzędnych okręgów

    # Iteracja po wykrytych okręgach i rysowanie ich na obrazie
    for i in circles[0, :]:
        # Rysowanie okręgu na obrazie (środek: (i[0], i[1]), promień: i[2])
        cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Wyświetlenie obrazu z narysowanymi okręgami
    cv2.imshow('obrazek', c_img)

# key k
def line():
    global image
    # Pobierz wartości progów dolnego i górnego z trackbarów
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    
    # Konwersja obrazu na skalę szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Wykrywanie krawędzi za pomocą algorytmu Canny'ego
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)
    
    # Wykrywanie linii za pomocą transformacji Hougha
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90,
                            minLineLength=100, maxLineGap=5)
    
    # Utworzenie kopii obrazu, aby narysować linie
    image_l = image.copy()
    
    # Iteracja po wykrytych liniach i rysowanie ich na obrazie
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Wyświetlenie obrazu z narysowanymi liniami
    cv2.imshow("obrazek", image_l)

# key o
def rotate():
    global image
    # Pobierz wartość kąta obrotu z trackbara o nazwie 'low'
    rot = cv2.getTrackbarPos('low', 'obrazek')
    
    # Pobierz wymiary obrazu
    height, width = image.shape[:2]
    
    # Oblicz środek obrazu
    center_x, center_y = (width / 2, height / 2)
    
    # Utwórz macierz transformacji dla obrotu obrazu
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    
    # Zastosuj macierz transformacji, aby obrócić obraz
    rotated_image = cv2.warpAffine(image, M, (width, height))
    
    # Wyświetl obrócony obraz w oknie o nazwie 'obrazek'
    cv2.imshow('obrazek', rotated_image)


def change_h(x):
    global fun
    if fun is not None:
        fun()



def main():
    global image, fun, files
    files = os.listdir(path_pliki)
    uploud(ord('0'))
    nimg = image.copy()
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
    # -----------wybor obrazka----------------
        if key >= ord('0') and key <= ord('9'):
            uploud(key)
            nimg = image.copy()
    # ----------------zmiana rozmiaru---------------
        elif key == ord('-'):
            resize()
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            cv2.imshow('obrazek', image)
            nimg = image.copy()
    # ----------------kolory------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        elif key == ord('r'):
            hsv_bitwais()
            fun = hsv_bitwais
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('z'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
    # ----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(image, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == ord('f'):
            morphology()
            fun = morphology
        elif key == ord('g'):
            morphology2()
            fun = morphology2
        elif key == ord('h'):
            marker()
            fun = marker
        elif key == ord('p'):
            connect_mask()
            fun = connect_mask
        elif key == ord('o'):
            rotate()
            fun = rotate
    # --------------------krztałty
        elif key == ord('j'):
            find_circle()
            fun = find_circle
        elif key == ord('k'):
            line()
            fun = line
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
