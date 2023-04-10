#!/usr/bin/env python3

def VoS(x):
  return x if isinstance(x,(str,int,float)) else x.name

def hline(s:str):
  s = s + "  "
  return f"% ----  {s:-<82}\n"

def cline(s:str):
  return f"% {s}\n"

def header(msg="Begin P file"):
  ls=f"% {'-'*88}\n"
  for s in msg.splitlines():
    ls = ls + cline(s)
  ls = ls + f"% {'-'*88}\n"
  return ls

def splicegen(maxchars, stringlist):
  """
  Return a list of slices to print based on maxchars string-length boundary.
  """
  runningcount = 0  # start at 0
  tmpslice = []  # tmp list where we append slice numbers.
  for item in stringlist:
    runningcount += len(item)
    if runningcount <= int(maxchars):
      tmpslice.append(item)
    else:
      yield tmpslice
      tmpslice = [item]
      runningcount = len(item)
  yield(tmpslice)

class MissingArgument(Exception):
  """Raised when arguments are missing"""
  def __init__(self, message="Missing arguments"):
    self.message = message
    super().__init__(self.message)


