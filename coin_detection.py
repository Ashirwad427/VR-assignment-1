import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def pre_img(img_p):
    img = cv2.imread(img_p)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clh = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(2, 2))
    clh_gry = clh.apply(gry)
    blr = cv2.GaussianBlur(clh_gry, (21, 21), 0)
    return img, gry, clh_gry, blr


def det_coin_hough(blr, img):
    circs = cv2.HoughCircles(
        blr, cv2.HOUGH_GRADIENT, dp=1.6, minDist=40, param1=100, param2=60, minRadius=125, maxRadius=185
    )
    hough_img = img.copy()
    if circs is not None:
        circs = np.uint16(np.around(circs))
        for i in circs[0, :]:
            cv2.circle(hough_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(hough_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return hough_img


def seg_coin(img):
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clh = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(2, 2))
    enh_gry = clh.apply(gry)
    blr = cv2.GaussianBlur(enh_gry, (21, 21), 0)
    _, bin_img = cv2.threshold(blr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    cls = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_coins = []
    out_img = np.zeros_like(img)
    for i, c in enumerate(cnts):
        msk = np.zeros_like(gry)
        cv2.drawContours(msk, [c], -1, (255), thickness=cv2.FILLED)
        col = np.random.randint(0, 255, (3,), dtype=int).tolist()
        out_img[msk == 255] = col
        x, y, w, h = cv2.boundingRect(c)
        seg_coins.append(img[y:y+h, x:x+w])
    return out_img, seg_coins



def save_img(img_p, out_d="coin_images"):
    os.makedirs(out_d, exist_ok=True)
    img, gry, clh_gry, blr = pre_img(img_p)
    hough_img = det_coin_hough(blr, img)
    out_img, seg_coins = seg_coin(img)
    canny = cv2.Canny(blr, 185, 110)


    cv2.imwrite(os.path.join(out_d, "hough_image.png"), hough_img)
    cv2.imwrite(os.path.join(out_d, "canny_image.png"), canny)
    cv2.imwrite(os.path.join(out_d, "segmented_coins_image.png"), out_img)

    for i, coin in enumerate(seg_coins):
        cv2.imwrite(os.path.join(out_d, f"coin_{i+1}.png"), coin)

    print(f"Imgs saved in '{out_d}'")
    print(f"Total coins detected: {len(seg_coins)}")


img_p = os.path.join(os.getcwd(), "coins.png")
save_img(img_p)
