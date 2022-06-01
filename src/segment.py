# a simple database to be used during early development
import math


def segment(roundtoo, fullsize, numfromfull):
    value = roundtoo/fullsize
    return math.ceil(numfromfull * value)
    

if __name__ == '__main__':
    print(segment(4,16,15))