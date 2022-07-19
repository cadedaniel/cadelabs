#!/usr/bin/env python3

import nltk
import pickle
import cloudpickle

class ObjectNoNtlk:
    def func(self):
        return 'return value!'

class ObjectYesNtlk:
    def func(self):
        return nltk.__version__

ObjectYesNtlk().func()

with open('no_ntlk_obj.pickle', 'wb') as f:
    f.write(cloudpickle.dumps(ObjectNoNtlk()))

with open('yes_ntlk_obj.pickle', 'wb') as f:
    f.write(cloudpickle.dumps(ObjectYesNtlk()))
