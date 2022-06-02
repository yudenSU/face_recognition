# a simple database to be used during early development
import math


# given the size of a the original frame, and a new size representing the number of buckets

def bucket(new_size, original_size, coor):
    value = new_size/original_size
    return math.ceil(coor[0] * value), math.ceil(coor[1] * value)
