#!/usr/bin/env python3

import nltk
import pickle

class DummyObject:
    pass

class DummyObject2:
    def func():
        pass

class DummyObject3:
    def func():
        return nltk.__version__

print(len(pickle.dumps(DummyObject)))
print(len(pickle.dumps(DummyObject())))

print(len(pickle.dumps(DummyObject2)))
print(len(pickle.dumps(DummyObject2())))

print(len(pickle.dumps(DummyObject3)))
print(len(pickle.dumps(DummyObject3())))
