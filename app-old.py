#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:14:59 2021

@author: telo
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello World'
