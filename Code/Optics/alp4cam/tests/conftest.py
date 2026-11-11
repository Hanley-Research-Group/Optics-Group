import sys
import types

import numpy as np
import pytest


class FakeALP4:
    def __init__(self, version="4.2", libDir=None):
        self.version = version
        self.libDir = libDir
        self.initialized = False
        self.seq_alloc = None
        self.timing = None
        self.run_loop = None
        self.imgData = None

    def Initialize(self):
        self.initialized = True

    def SeqAlloc(self, nbImg, bitDepth):
        self.seq_alloc = (nbImg, bitDepth)

    def SeqPut(self, imgData):
        self.imgData = imgData

    def SetTiming(self, pictureTime=None, illuminationTime=None, synchDelay=None, synchPulseWidth=None, triggerInDelay=None):
        self.timing = {
            "pictureTime": pictureTime,
            "illuminationTime": illuminationTime,
            "synchDelay": synchDelay,
            "synchPulseWidth": synchPulseWidth,
            "triggerInDelay": triggerInDelay,
        }

    def Run(self, loop=True):
        self.run_loop = loop

    def Halt(self):
        return None

    def FreeSeq(self):
        return None

    def Free(self):
        self.initialized = False


@pytest.fixture(autouse=True)
def fake_hardware(monkeypatch):
    fake_alp_module = types.ModuleType("ALP4")
    fake_alp_module.ALP4 = FakeALP4
    monkeypatch.setitem(sys.modules, "ALP4", fake_alp_module)
