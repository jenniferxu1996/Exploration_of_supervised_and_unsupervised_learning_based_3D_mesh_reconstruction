#!/bin/bash

CLASS="02958343"

for file in /home/grisw/Downloads/ShapeNetCore.v2/$CLASS/*
do
    python generate_data.py -a $CLASS -b $file
done

