
print("Hello World")

#import libcpp.cstdlib as cstdlib

#from cython.cimports.libc.stdlib import atoi
from libc.stdlib cimport abort

def call_abort():
    abort()

#from libcpp.cstdlib cimport abort
