# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:12:41 2021

@author: 17832020
"""
import os
import glob
import csv
import xlwt

for csvfile in glob.glob(os.path.join('.', '*.csv')):
    print("starting new workbook")
    wb = xlwt.Workbook()
    fpath = csvfile.split("/", 1)
    fname = fpath[1].split(".", 1) ## fname[0] should be our worksheet name
    print("adding sheet " + fname[0])
    ws = wb.add_sheet(fname[0])
    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            for c, col in enumerate(row):
                ws.write(r, c, col)
            print("saving workbook")
    wb.save('Combined_sheets.xls')