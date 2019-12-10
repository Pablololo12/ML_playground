#!/usr/bin/env python3
# Copyright (c) 2019, ARM Limited and Contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import cv2
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image extracter")
    parser.add_argument('-f', help="Video file", type=str, required=True)
    parser.add_argument('-d', help="Directory to save", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()

    cap = cv2.VideoCapture(args.f)
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if i%10 == 0:
            cv2.imwrite(os.path.join(args.d,'pic_'+str(i)+'.jpg'), frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()