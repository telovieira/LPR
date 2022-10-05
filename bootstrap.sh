#!/bin/bash
/root/.profile
export FLASK_APP=/root/license_plates/app.py
flask run --host 0.0.0.0 --port=8088
