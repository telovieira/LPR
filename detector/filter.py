#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:14:11 2021

@author: telo
"""

import cv2


def filter_mask(img, a=None):
    '''
        This filters are hand-picked just based on visual tests
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=1)

    return dilation