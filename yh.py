#!/usr/bin/env python
import aiohttp
import argparse
import asyncio
import cv2
from imutils import contours
import imutils
import numpy as np
import os
import pytesseract
from shutil import copyfile
from skimage import exposure
import json
import requests
import re
import sys

class Card(object):
    @classmethod
    async def create(cls, bid, cid, name=None, value=None):
        self = Card()
        self.bid = bid
        self.cid = cid
        if name and value:
            self.name = name
            self.value = value
        else:
            cid, name, value = await api_get_price(self.cid)
            if cid is not None:
                self.name = name
                self.value = value
        return self

    def __str__(self):
        return '{},{},{},{:.2f}\n'.format(
            self.bid,self.cid,self.name,self.value)

    def save(self, fn='cards.csv'):
        #header = "bid,cid,name,value\n"
        header = ""
        if os.path.exists(fn):
            copyfile(fn, fn+".bak")
            header = ""
        with open(fn, 'a') as csv_file:
            csv_file.write(header+str(self))


def cid_from_image(image):
    W = 800

    # Size regularization
    ratio = image.shape[1] / W
    orig = image.copy()
    image = imutils.resize(image, width=W)

    # Blur + edge detect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # Get contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return ""

    # Grab corners and warp
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    rect *= ratio

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    w, h, _ = warp.shape
    hw, hh = w // 2, h // 2
    if not 1.2 < w/h < 1.8:
        return ""
    warp = imutils.resize(warp, width=2000)
    warp = warp[-900:-500,-600:]
    if not warp.any():
        return ""


    # Identify areas to search
    a, b = 11, 1
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9*a, 3*b))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))

    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    if maxVal - minVal == 0:
        return ""
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    loc = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / h
        if 4 < ar < 7 and 100 < w < 400:
            loc = (x,y,w,h)
            cv2.rectangle(warp,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(warp, 'x:{} y:{}'.format(x,y),
                        (x,y-10), 0, 0.3, (0,255,0))

    if not loc:
        return ""

    (gX,gY,gW,gH) = loc
    pad = 15
    group = gray[gY - pad:gY + gH + pad, gX - pad:gX + gW + pad]
    if not group.any():
        return ""
    group = cv2.adaptiveThreshold(group,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,39,8)

    show = [group]
    #cv2.imshow("", np.hstack(show))
    t = pytesseract.image_to_string(group)
    return t.strip()


def ilogging(fname):
    image = cv2.imread(fname)
    text = cid_from_image(image)
    if text:
        print(text)
    else:
        print("Could not find code")

async def vlogging():
    bid = int(input("Enter book number that these cards are sorted in > ").strip())
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*4)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*4)

    while(True):
        ret, image = cap.read()

        text = cid_from_image(image)
        if text:
            if re.match("^([A-Z]{2}|[A-Z]{3}|[A-Z]{4})-[A-Z]*[0-9]{3}$", text):
                cid, name, val = await api_get_price(text)
                if cid is not None:
                    va = input("Is this \"{}\"? [Yn] > ".format(name)).strip().upper()
                    if va == "Q":
                        break
                    if va != "N":
                        print("Adding {}, worth ${:.2f}".format(name, val))
                        card = await Card.create(bid, cid, name, val)
                        card.save()

    cap.release()
    cv2.destroyAllWindows()



async def logging():
    print("In logging mode")
    bid = int(input("Enter book number that these cards are sorted in > ").strip())

    print("Entering main loop, enter \"Q\" to quit...")
    while True:
        cid = input("Enter the part name > ").strip().upper()
        if cid == "Q":
            break
        va = input("Is \"{}\" valid? [Yn] > ".format(cid)).strip().upper()
        if va == "N":
            print("Not adding \"{}\"".format(cid))
            continue
        card = await Card.create(bid, cid)
        card.save()

async def api_get_price(pn):
    url = "http://yugiohprices.com/api/price_for_print_tag/{}".format(pn)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                text = json.loads(await resp.text())
                if text['status'] == 'fail':
                    return (None, None, None)

                name = text['data']['name']
                avg = text['data']['price_data']['price_data']\
                    ['data']['prices']['average']
                return (pn, name, avg)

    return (None, None, None)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save some cards')
    parser.add_argument("--image-log", "-i",
                        default=False,
                        help="Card logging mode")
    parser.add_argument("--video-log", "-v", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Card logging mode")
    parser.add_argument("--log", "-l", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Card logging mode")

    args = parser.parse_args()

    image_log = args.image_log
    video_log = args.video_log
    log = args.log

    if sum([video_log, log]) > 1:
        raise Exception('Please pass only one option per run')

    if image_log:
        print(image_log)
        ilogging(image_log)
        sys.exit(0)

    if video_log:
        asyncio.run(vlogging())
        sys.exit(0)

    if log:
        asyncio.run(logging())
        sys.exit(0)

    print("Nothing to do")
    sys.exit(1)

