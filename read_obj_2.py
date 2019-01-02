#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:59:46 2018

@author: Davide
"""

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from io import open

import numpy as np

#import sys # von mir hinzugefügt Alex
#np.set_printoptions(threshold = sys.maxsize) # von mir hinzugefügt Alex

def read_obj_file(file_name):
    ''' Read an obj file

    input
        file_name: the obj file path to read

    output
        vertices_array: np.array ov vertices
        faces_array: np.array of face triangular faces
    '''
    #--------------------------------------------------------------------------
    #                                notes
    #--------------------------------------------------------------------------
    #    1) in this implementation a line wrap in obj file caure an error.
    #       This problem can be fixed
    #    2) for irregular meshes, faces can be stored as list of lists.
    #       For that, change the output from np.array(faces_list) to
    #       faces_list directly
    #--------------------------------------------------------------------------

    objFile = open(file_name, encoding='utf-8')
    verticesList = []
    facesList = []
    for l in objFile:
        splitedLine = l.split(' ')
        if splitedLine[0] == 'v':
            try:
                splitX = splitedLine[1].split('\n')
                x = float(splitX[0])
                splitY = splitedLine[2].split('\n')
                y = float(splitY[0])
                splitZ = splitedLine[3].split('\n')
                z = float(splitZ[0])
            except ValueError:
                print('Disable line wrap when saving .obj!')
            verticesList.append([x, y ,z])
        elif splitedLine[0] == 'f':
            vList = []
            L = len(splitedLine)
            try:
                for i in range(1, L):
                    splitedFaceData = splitedLine[i].split('/')
                    vList.append(int(splitedFaceData[0]) - 1 )
                facesList.append(vList)
            except ValueError:
                vList = []
                for i in range(1, L-1):
                    vList.append(int(splitedLine[i]) - 1 )
                facesList.append(vList)
    vertices_array = np.array(verticesList, dtype='f')
    faces_array = np.array(facesList, dtype='i')
    return vertices_array, faces_array


#------------------------------------------------------------------------------
#                                  Test
#------------------------------------------------------------------------------
# this works for obj files in the same folder of the script, otherwise the
# full path should be specified
#------------------------------------------------------------------------------

import os

path = os.path.dirname(os.path.abspath(__file__))

file_name = path + '\\part1_moved.obj'
file_name2 = path + '\\part2_moved.obj'
file_name3 = path + '\\part3_moved.obj'
file_name4 = path + '\\part4_moved.obj'
file_name5 = path + '\\part5_moved.obj'
file_name6 = path + '\\part6_moved.obj'
file_name7 = path + '\\reference.obj'

P, F = read_obj_file(file_name)
P2, F = read_obj_file(file_name2)
P3, F = read_obj_file(file_name3)
P4, F = read_obj_file(file_name4)
P5, F = read_obj_file(file_name5)
P6, F = read_obj_file(file_name6)
#print(P)
#print(F)  # von mir wegkommentiert (Alex)

X, F_moved = read_obj_file(file_name7)