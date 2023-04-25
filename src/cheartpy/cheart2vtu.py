#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pvpython script to convert CHeart data to vtk unstructured grid format (vtu)
#
# author: Andreas Hessenthaler
# modified by: Will Zhang (willwz@gmail.com)
# data: 3/31/2022
from __future__ import print_function
import os.path
import numpy as np
import numpy.typing as npt
import sys
import argparse
import glob
import re
import struct
import enum
import dataclasses
import typing as tp

from .meshing.vtk_elements import *

################################################################################################
# Compressed binary VTU file format?
use_Compression = True
################################################################################################
# The argument parse
parser = argparse.ArgumentParser(description='converts cheart output Dfiles into vtu files for paraview')
parser.add_argument('--prefix',        '-p', dest='prefix',      action='store', default='paraview',
                    type=str, help='OPTIONAL: supply a prefix name to be used for the exported vtu files. If -p is not supplied, then "paraview" will be used. that is the outputs will be named paraview-#.D')
parser.add_argument('--folder',        '-f', dest='infolder',     action='store', default='',
                    type=str, help='OPTIONAL: supply the path to the folder where the .D files are stored. If -f is not supplied, then the path is assumed to be the current folder.')
parser.add_argument('--out-folder',    '-o', dest='outfolder',    action='store', default='',
                    type=str, help='OPTIONAL: supply the path to the folder where the vtu outputs should be saved to. If -f is not supplied, then the path is assumed to be the current folder.')
parser.add_argument('--index',         '-i', nargs=3, dest='irange',    action='store', default=[0,0,1],
                    type=int, metavar=('start', 'end', 'step'), help='MANDATORY: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.')
parser.add_argument('--use-subiter',   '-si', dest='subiter', nargs=3, action='store', default=None,
                    type=int, metavar=('start', 'end', 'step'), help='OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.')
parser.add_argument('--use-subauto',   '-sa', dest='subauto', action='store_true',
                    help='OPTIONAL: specify the start, end, and step for the range of data files. If -i is not used, only step 0 will be processed. For non trivial use, this is mandatory.')
topgroup = parser.add_argument_group(title='Topology')
topgroup.add_argument('--space',         '-x', dest='xfile',        action='store', default="mesh_FE.X",
                    type=str, help='MANDATORY: supply a relative path and file name  from the current directory to the .X file (must end in .X), or a variable name relative to the input folder for the Space variable. default is "Space" in cuurent folder')
topgroup.add_argument('--t-file',        '-t', dest='tfile',        action='store', default="mesh_FE.T",
                    type=str, help='MANDATORY: supply a relative path and file name from the current directory to the topology file, the default is mesh_FE.T')
topgroup.add_argument('--b-file',        '-b', dest='bfile',        action='store', default=None,
                    type=str, help='OPTIONAL: supply a relative path and file name  from the current directory to the boundary file, the default is None')
settinggroup = parser.add_argument_group(title='Settings')
settinggroup.add_argument('--no-progressbar',   action='store_false', dest='progress',
                          help='OPTIONAL: controls whether to show a progress bar. Default is True.')
settinggroup.add_argument('--verbose', '-v', action='store_true',
                          help='OPTIONAL: print more info')
settinggroup.add_argument('--binary', action='store_true',
                          help='OPTIONAL: assumes that the .D files being imported is binary')
settinggroup.add_argument('--no-compression', '-n', dest='compress',action='store_false',
                          help='OPTIONAL: disable compression.')
settinggroup.add_argument('--cores', '-c', dest='cores', action='store', default=None, type=int, metavar=('#'),
                          help='OPTIONAL: use multicores.')
parser.add_argument('--make-time-series', dest='time_series', default=None,
                         type=str, help='OPTIONAL: incorporate time data, supply a file for the time step.')
subparsers = parser.add_subparsers(help='Collective of subprogram modules')
parser_find= subparsers.add_parser('find', help='determine settings automatically')
parser_find.add_argument('--mesh', dest='mesh', action='store', default='',
                         type=str, help='OPTIONAL: supply a directory which contains the mesh files')
parser_find.add_argument('--step', dest='mesh', action='store', default=1,
                         type=str, help='OPTIONAL: supply a directory which contains the mesh files')


parser.add_argument('variablenames', nargs='*', action='store', default=list(),
                    type=str, metavar=('var'), help='Optional: specify the variables to add to the vtu files. Multiple variable can be listed consecutively.')
################################################################################################
# Get a library to use for compression
try:
  from paraview.simple import XMLUnstructuredGridReader, XMLUnstructuredGridWriter
  compress_method = 'pvpython'
except ImportError:
  try:
    from vtk import VTK_MAJOR_VERSION, vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
    compress_method = 'vtk'
  except ImportError:
    try:
      import meshio
      compress_method = 'meshio'
    except ImportError:
      use_Compression = False
      print(">>>WARNING: You need either python-paraview or python-vtk or meshio for compression. Compression option disabled.")
################################################################################################
# Check if multiprocessing is available
try:
  from concurrent import futures
  futures_avail = True
except:
  futures_avail = False
################################################################################################

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', printEnd = "\r"):
  """
  Call in a loop to create terminal progress bar
  @params:
      iteration   - Required  : current iteration (Int)
      total       - Required  : total iterations (Int)
      prefix      - Optional  : prefix string (Str)
      suffix      - Optional  : suffix string (Str)
      decimals    - Optional  : positive number of decimals in percent complete (Int)
      length      - Optional  : character length of bar (Int)
      fill        - Optional  : bar fill character (Str)
      printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
  """
  percent = ("{0:." + str(decimals) + "f}").format(100 if (total == 0) else 100 * (iteration / float(total)))
  filledLength = int(length if (total == 0) else length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
  # Print New Line on Complete
  if iteration == total:
      print()

class progress_bar:
  def __init__(self, message, max=100):
    self.n = max
    self.i = 0
    self.message = message
    printProgressBar(self.i, self.n, prefix = self.message, suffix = 'Complete', length = 50)
  def next(self):
    self.i = self.i + 1
    printProgressBar(self.i, self.n, prefix = self.message, suffix = 'Complete', length = 50)
  def finish(self):
    printProgressBar(self.n, self.n, prefix = self.message, suffix = 'Complete', length = 50)

################################################################################################
# Codes
################################################################################################
# set file suffixes for VTU and CHeart (and possible compression)
vtksuffix           = ".vtu"
cheartsuffix        = ".D"
gzsuffix            = ".gz"

def read_D_binary(file) -> np.ndarray:
  with open(file, mode='rb') as f:
    nnodes = struct.unpack("i", f.read(4))[0]
    dim    = struct.unpack("i", f.read(4))[0]
    arr    = np.zeros((nnodes, dim))
    for i in range(nnodes):
      for j in range(dim):
        bite     = f.read(8)
        if (not bite): raise BufferError('Binary buffer being read ran out before indicated range')
        arr[i,j] = struct.unpack("d", bite)[0]
  return arr


def print_header() ->None:
  print("################################################################################################")
  print("")
  print("    pvpython script to convert CHeart data to vtk unstructured grid format (vtu)")
  print("")
  print("    author: Andreas Hessenthaler")
  print("    modified by: Will Zhang")
  print("    data: 3/31/2022")
  print("")
  print("################################################################################################")


def print_input_info(args) -> None:
  print("The output will be saved to ", args.outfolder)
  print("The file name of the output will start with", args.prefix)
  print("The script will be ran through the following range", args.irange)
  print("The data are stored in the folder ", args.infolder)
  print("The topology file to use is ", args.tfile)
  print("The boundary file to use is ", args.bfile)
  print("The space file to use is ", args.xfile)
  print("The varibles to add are: ")
  print(args.variablenames)
  print("<<< Output folder:               {}".format(args.outfolder))
  print("<<< Output file name prefix:     {}".format(args.prefix))

class CheartDataFormat(enum.Enum):
  mesh = 0
  var  = 1
  zip  = 2


def compress_vtu(name:str, method:str = 'meshio', verbose:bool=False) -> None:
  if method=='pvpython':
    # read ASCII vtu file and write raw vtu file
    fin = XMLUnstructuredGridReader(FileName=name)
    # CompressorType: 0 - None, 1 - ZLib, 2 - LZ4, 3 - LZMA
    # DataMode: 0 - Ascii, 1 - Binary, 2 - Appended
    foutbin = XMLUnstructuredGridWriter(CompressorType=1, DataMode=2)
    foutbin.FileName = name
    foutbin.UpdatePipeline()
  elif method=='vtk':
    vtuReader=vtkXMLUnstructuredGridReader()
    vtuReader.SetFileName(name)
    vtuReader.Update()
    vtuWriter=vtkXMLUnstructuredGridWriter()
    vtuWriter.SetDataModeToAppended()
    vtuWriter.SetCompressorTypeToZLib()
    vtuWriter.SetFileName(name)
    if VTK_MAJOR_VERSION <= 5:
        vtuWriter.SetInput(vtuReader.GetOutput())
    else:
        vtuWriter.SetInputData(vtuReader.GetOutput())
    vtuWriter.Write()
  elif method=='meshio':
    if verbose:
      size = os.stat(name).st_size
      print("File size before: {:.2f} MB".format(size / 1024 ** 2))
    mesh = meshio._helpers.read(name, file_format='vtu')
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    if verbose:
      size = os.stat(name).st_size
      print("File size after: {:.2f} MB".format(size / 1024 ** 2))
  else:
    raise ImportError("How did you get here! Should have found that none of the module for compression was found!")


@dataclasses.dataclass
class InputArgs:
  vars : tp.List[str]
  i0 : int = 0
  it : int = 1
  di : int = 1
  prefix : str = 'paraview'
  outputfile : str = 'paraview'
  topologyname : str = "Mesh_FE.T"
  boundaryname : tp.Optional[str] = None
  spacename : str = "Mesh_FE.X"
  deformedSpace : tp.Optional[str] = None
  progress : bool = False
  useCompression : bool = False
  binary : bool = False
  verbose : bool = False
  cores : int = 1

  spacetype : CheartDataFormat = CheartDataFormat.mesh
  spacetemp : tp.Callable[[int], str] = lambda x : str(x)
  varstype : tp.List[CheartDataFormat] = dataclasses.field(default_factory=list)
  varstemp : tp.List[tp.Callable[[int], str]] = dataclasses.field(default_factory=list)

def get_inputs(args) -> InputArgs:
  print(args)
  inp = InputArgs(args.variablenames)

  inp.i0        = args.irange[0]
  inp.it        = args.irange[1] + 1
  inp.di        = args.irange[2]

  inp.infolder  = args.infolder
  if args.outfolder != '':
    if not os.path.isdir(args.outfolder):
      print(">>>WARNING: output directory does not exist, it will be created!")
    os.makedirs(args.outfolder, exist_ok=True)
  inp.prefix    = args.prefix
  inp.outputfile = os.path.join(args.outfolder, args.prefix)

  inp.spacename    = args.xfile
  inp.topologyname = args.tfile
  inp.boundaryname = args.bfile
  if args.bfile is None:
    print(">>>NOTICE: No boundary file is supplied ")

  inp.deformedSpace       = inp.spacename.split("+")
  if (len(inp.deformedSpace) == 2):
    inp.spacename         = inp.deformedSpace[0]
  inp.deformedSpace       = os.path.join(inp.infolder, inp.spacename).split("+")
  ################################################################################################
  # get run path and add trailing slash if it's not already there
  # !!! NOTE(WILL): check how the spacename works
  inp.progress = False if args.verbose else args.progress
  inp.useCompression = args.compress if use_Compression else use_Compression
  inp.binary = args.binary
  inp.verbose = args.verbose
  if args.cores is not None:
    if futures_avail:
      inp.cores = args.cores
    else:
      raise ModuleNotFoundError("Concurrent.futures is not available for parallel processing please turn off")
  else:
    inp.cores = None
  return inp


def check_variables(inp : InputArgs) -> None:
  filecheckerr = False
  ################################################################################################
  # Check all requested items on whether they exists
  print("")
  print("<<< Time series:                 From {} to {} with increment of {}".format(inp.i0, inp.it - 1, inp.di))
  print("<<< Compressed VTU format:       {}".format(compress_method if inp.useCompression else "None"))
  print("<<< Import data as binary:       {}".format(inp.binary))
  if inp.boundaryname is not None:
    print("<<< Output file name (boundary): {}_boundary{}".format(inp.prefix, vtksuffix))
  # pre-flight to check whether all files exist
  if not(os.path.isfile(inp.topologyname)):
    print(">>>ERROR: File "+inp.topologyname+" does not exist.")
    filecheckerr = True
  if inp.boundaryname is not None and not(os.path.isfile(inp.boundaryname)):
    print(">>>WARNING: File "+inp.boundaryname+" does not exist.")
  ################################################################################################
  # check what the space name is
  if os.path.isfile(inp.spacename):
    inp.spacetype = CheartDataFormat.mesh
    inp.spacetemp = lambda i: inp.spacename
  elif os.path.isfile(os.path.join(inp.infolder, f"{inp.spacename}-{inp.i0}{cheartsuffix}")):
    inp.spacetype = CheartDataFormat.var
    inp.spacetemp = lambda i: os.path.join(inp.infolder, f"{inp.spacename}-{i}{cheartsuffix}")
  elif os.path.isfile(os.path.join(inp.infolder, f"{inp.spacename}-{inp.i0}{cheartsuffix}.gz")):
    inp.spacetype = CheartDataFormat.zip
    inp.spacetemp = lambda i: os.path.join(inp.infolder, f"{inp.spacename}-{i}{cheartsuffix}.gz")
  else:
    print(f">>>ERROR: No file matching the template {inp.spacename} can be found.")
    filecheckerr = True
  ################################################################################################
  # check what the space name is
  for var in inp.vars:
    if os.path.isfile(os.path.join(inp.infolder, f"{var}-{inp.i0}{cheartsuffix}")):
      inp.varstype.append(CheartDataFormat.var)
      inp.varstemp.append(lambda v, i: os.path.join(inp.infolder, f"{v}-{i}{cheartsuffix}"))
    elif os.path.isfile(os.path.join(inp.infolder, f"{var}-{inp.i0}{cheartsuffix}.gz")):
      inp.varstype.append(CheartDataFormat.zip)
      inp.varstemp.append(lambda v, i: os.path.join(inp.infolder, f"{v}-{i}{cheartsuffix}.gz"))
    else:
      print(f">>>ERROR: No file matching the template for {var} at time step {inp.i0} can be found.")
      filecheckerr = True
  ################################################################################################
  # loop over all requested times
  lastvariable = [None for _ in inp.vars]
  warnvariable = [True for _ in inp.vars]
  for time in range(inp.i0, inp.it, inp.di):
    for i, var in enumerate(inp.vars):
      if not(os.path.isfile(inp.varstemp(var, time))):
        print(f">>>WARNING: File {var} at step {time} not found. Using file from previous step.")
        warnvariable[i] = False
  if filecheckerr:
    raise FileNotFoundError(">>>FILE ERROR!!!")


class LoadTopology:
  def __init__(self, inp: InputArgs) -> None:
    ################################################################################################
    # read topology and get number of elements, number of nodes per elements
    self._ft  = np.loadtxt(inp.topologyname, skiprows=1, dtype=int)
    if self._ft.ndim==1:
      self._ft = np.array([self._ft])
    self.ne = self._ft.shape[0]
    self.nc = self._ft.shape[1]
    # guess the VTK element type
    # bilinear triangle
    if (self.nc == 3):
      self.vtkelementtype = VtkBilinearTriangle
      self.vtksurfacetype = VtkLinearLine
    # biquadratic triangle
    elif (self.nc == 6):
      self.vtkelementtype = VtkBiquadraticTriangle
      self.vtksurfacetype = VtkQuadraticLine
    # bilinear quadrilateral / trilinear tetrahedron
    elif (self.nc == 4):
      if inp.boundaryname is None:
        print(">>>ERROR: bilinear quadrilateral / trilinear tetrahedron detected but boundary file supplied is ", inp.boundaryname)
        print("          please supply the corresponding boundary file to proceed ", inp.boundaryname)
        sys.exit()
      else:
        with open(inp.boundaryname, "r") as f:
          _ = f.readline().strip()
          second_line = [int(i) for i in f.readline().strip().split()]
        # bilinear quadrilateral
        if (len(list(second_line)) == 4):
          self.vtkelementtype = VtkBilinearQuadrilateral
          self.vtksurfacetype = VtkLinearLine
        # trilinear tetrahedron
        elif (len(list(second_line)) == 5):
          self.vtkelementtype = VtkTrilinearTetrahedron
          self.vtksurfacetype = VtkBilinearTriangle
        else:
          print(f">>>ERROR: boundary file {inp.boundaryname} is not consistent with the topology file {inp.topologyname}")
          sys.exit()
    # biquadratic quadrilateral
    elif (self.nc == 9):
      self.vtkelementtype = VtkBiquadraticQuadrilateral
      self.vtksurfacetype = VtkQuadraticLine
    # triquadratic tetrahedron
    elif (self.nc == 10):
      self.vtkelementtype = VtkTriquadraticTetrahedron
      self.vtksurfacetype = VtkQuadraticLine
    # trilinear hexahedron
    elif (self.nc == 8):
      self.vtkelementtype = VtkTrilinearHexahedron
      self.vtksurfacetype = VtkBilinearQuadrilateral
    # triquadratic hexahedron
    elif (self.nc == 27):
      self.vtkelementtype = VtkTriquadraticHexahedron
      self.vtksurfacetype = VtkBiquadraticQuadrilateral
    else:
      sys.exit(">>>ERROR: Element type not implemented.")
  def __setitem__(self, index, data):
      self._ft[index] = data
  def __getitem__(self, index):
      return self._ft[index]


def export_boundary(tp : LoadTopology, inp : InputArgs) -> None:
  ################################################################################################
  # export patch IDs from B-file
  if inp.boundaryname is None:
    print(">>> NOTICE: No boundary file is supplied, export of boundary patch is skipped")
  else:
    print("<<< Working on exporting the boundary patch IDs from", inp.boundaryname)
    # for boundary coordinates, let's default to first time step
    time = inp.i0
    # read boundary and get number of elements, number of nodes per elements
    fb      = np.loadtxt(inp.boundaryname, skiprows=1, dtype=int)
    fb = fb[:,1:]
    fben    = fb.shape[0]
    fbcn    = fb.shape[1] - 1
    # write header
    foutfile = inp.outputfile+"_boundary"+vtksuffix
    # print(foutfile)
    with open(foutfile, "w") as fout:
      # convert space variable
      fx  = np.loadtxt(inp.spacetemp(0),  skiprows=1, dtype=float)
      fnn = fx.shape[0]
      fnd = fx.shape[1]
      # if deformed space, then add displacement
      if (len(inp.deformedSpace) == 2):
        dfile   = inp.infolder + inp.deformedSpace[1] + "-" + str(time) + ".D"
        if not(os.path.isfile(dfile)):
          dfile = dfile + ".gz"
        fd  = np.loadtxt(dfile, skiprows=1, dtype=float)
        fx  = fx + fd
      # VTU files are defined in 3D space, so we have to append a zero column for 2D data
      if (fnd == 1):
        print(">>>ERROR: Cannot convert data that lives on 1D domains.")
        return
      elif (fnd == 2):
        z   = np.zeros((fnn,1), dtype=float)
        fx  = np.append(fx, z, axis=1)
      # start
      fout.write(f'<VTKFile type="UnstructuredGrid">\n')
      fout.write(f'{" "*2}<UnstructuredGrid>\n')
      fout.write(f'{" "*4}<Piece Name="{inp.prefix}" NumberOfPoints="{fnn}" NumberOfCells="{fben}">\n')
      # boundary points
      fout.write(f'{" "*6}<Points>\n')
      fout.write(f'{" "*8}<DataArray type="Float64" NumberOfComponents="3" Format="ascii">\n')
      for points in range(fx.shape[0]):
        fout.write(f'{" "*10}{fx[points, 0]:.16f} {fx[points, 1]:.16f} {fx[points, 2]:.16f}\n')
      fout.write(f'{" "*8}</DataArray>\n')
      fout.write(f'{" "*6}</Points>\n')
      # boundary patch ids
      fout.write(f'{" "*6}<CellData Scalars="scalars">\n')
      fout.write(f'{" "*8}<DataArray type="Int8" Name="PatchIDs" Format="ascii">\n')
      for patchIdx in range(0, fben):
        fout.write(f'{" "*10}{fb[patchIdx, - 1]:i}\n')
      fout.write(f'{" "*8}</DataArray>\n')
      fout.write(f'{" "*6}</CellData>\n')
      # boundary topology
      fout.write(f'{" "*6}<Cells>\n')
      fout.write(f'{" "*8}<DataArray type="Int64" Name="connectivity"  Format="ascii">\n')
      for elem in range(0, fben):
        tp.vtksurfacetype.write(fout=fout, elem=fb[elem])
      fout.write(f'{" "*8}</DataArray>\n')
      # boundary locations
      fout.write(f'{" "*8}<DataArray type="Int64" Name="offsets"  Format="ascii">\n')
      for points in range(fbcn, (fbcn)*(fben+1), fbcn):
        fout.write(f'{" "*10}{points:i}')
      fout.write(f'\n')
      fout.write(f'{" "*8}</DataArray>\n')
      # boundary types
      fout.write(f'{" "*8}<DataArray type="Int8" Name="types"  Format="ascii">\n')
      for points in range(fben):
        fout.write(f'{" "*10}{tp.vtkelementtype.vtksurfaceid}\n')
      fout.write(f'{" "*8}</DataArray>\n')
      fout.write(f'{" "*6}</Cells>\n')
      # end
      fout.write(f'{" "*4}</Piece>\n')
      fout.write(f'{" "*2}</UnstructuredGrid>\n')
      fout.write(f'</VTKFile>\n')
    if inp.useCompression:
      compress_vtu(foutfile, method=compress_method, verbose=inp.verbose)


def export_data_iter(tp : LoadTopology, inp : InputArgs, time, lastvariable, lastspace, lastdeform) -> None:
  # print(f'bart is {bart}')
  ############################################################################################
  # convert space variable
  xfile = inp.spacetemp(time)
  if os.path.isfile(xfile):
    lastspace = xfile
  elif lastspace is None:
    sys.exit(">>>ERROR: Even though the space file was found during check, it is not found during import")
  else:
    xfile = lastspace

  fx  = np.loadtxt(inp.spacetemp(time),  skiprows=1, dtype=float)
  fnn = fx.shape[0]
  fnd = fx.shape[1]
  # if deformed space, then add displacement
  if (len(inp.deformedSpace) == 2):
    dfile   = inp.infolder + inp.deformedSpace[1] + "-" + str(time) + ".D"
    if not(os.path.isfile(dfile)):
      dfile = dfile + ".gz"
      if not(os.path.isfile(dfile)):
        if lastdeform is None:
          sys.exit(">>>ERROR: Even though the deformed space file was found during check, it is not found during import")
        else:
          dfile = lastdeform
      else:
        lastdeform = dfile
    else:
      lastdeform = dfile
    fd  = np.loadtxt(dfile, skiprows=1, dtype=float)
    fx  = fx + fd
  # VTU files are defined in 3D space, so we have to append a zero column for 2D data
  if (fnd == 1):
    print(">>>ERROR: Cannot convert data that lives on 1D domains.")
    return
  elif (fnd == 2):
    z   = np.zeros((fnn,1), dtype=float)
    fx  = np.append(fx, z, axis=1)

  foutfile = inp.outputfile+"-"+str(time)+vtksuffix

  with open(foutfile, "w") as fout:
    fout.write("<VTKFile type=\"UnstructuredGrid\">\n")
    fout.write("  <UnstructuredGrid>\n")
    fout.write("    <Piece Name=\""+inp.prefix+"\" NumberOfPoints=\""+str(fnn)+"\" NumberOfCells=\""+str(tp.ne)+"\">\n")
    # space
    fout.write("      <Points>\n")
    fout.write("        <DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
    for points in range(fx.shape[0]):
      fout.write("          % .16f % .16f % .16f\n" % (fx[points, 0], fx[points, 1], fx[points, 2]))
    fout.write("        </DataArray>\n")
    fout.write("      </Points>\n")
    ############################################################################################
    # convert other variables
    fout.write("      <PointData Scalars=\"scalars\">\n")
    for varIdx in range(0, len(inp.vars), 1):
      vfile = inp.varstemp[varIdx](time)
      if not(os.path.isfile(vfile)):
        if lastvariable[varIdx] is not None:
          vfile = lastvariable[varIdx]
        else:
          sys.exit(">>>ERROR: Unexpect problem, file was found during check but not when being imported: {}".format(lastvariable[varIdx]))
      else:
        lastvariable[varIdx] = vfile
      # print(f'file type is {inp.binary}')
      if inp.binary:
        fv = read_D_binary(vfile)
        # print(fv.shape)
      else:
        fv = np.loadtxt(vfile,  skiprows=1, dtype=float)
      fvnn = fv.shape[0]
      if (fv.ndim == 1):
        fvnd = 1
      else:
        fvnd = fv.shape[1]
      # check whether number of scalars matches number of spatial nodes
      if (fnn != fvnn):
        raise AssertionError(">>>ERROR: Invalid number of nodes for "+str(inp.vars)+".")
      # append zero column if need be
      if (fvnd == 2):
        z       = np.zeros((fvnn,1), dtype=float)
        fv      = np.append(fv, z, axis=1)
        fvnd    = 3
      fout.write("        <DataArray type=\"Float64\" Name=\""+str(inp.vars[varIdx])+"\" NumberOfComponents=\""+str(fvnd)+"\" Format=\"ascii\">\n")
      stringFormat    = " % .16f"*fvnd
      stringFormat    = "         " + stringFormat + "\n"
      if(fvnd == 1):
        for points in range(fnn):
          fout.write(stringFormat % (fv[points]))
      else:
        for points in range(fnn):
          fout.write(stringFormat % tuple(fv[points, 0:fvnd:1]))
      fout.write("        </DataArray>\n")
    fout.write("      </PointData>\n")
    ############################################################################################
    # topology
    fout.write("      <Cells>\n")
    fout.write("        <DataArray type=\"Int64\" Name=\"connectivity\"  Format=\"ascii\">\n")
    for elements in range(0, tp.ne):
      tp.vtkelementtype.write(fout, elem=tp[elements])
    fout.write("        </DataArray>\n")
    # cell locations
    fout.write("        <DataArray type=\"Int64\" Name=\"offsets\"  Format=\"ascii\">\n")
    for points in range(tp.nc, tp.nc*(tp.ne+1), tp.nc):
      fout.write("          %i" % (points))
    fout.write('\n')
    fout.write("        </DataArray>\n")
    # cell types
    fout.write("        <DataArray type=\"Int8\" Name=\"types\"  Format=\"ascii\">\n")
    for points in range(tp.ne):
      fout.write("          "+str(tp.vtkelementtype)+"\n")
    fout.write("        </DataArray>\n")
    fout.write("      </Cells>\n")

    # end
    fout.write("    </Piece>\n")
    fout.write("  </UnstructuredGrid>\n")
    fout.write("</VTKFile>\n")
  if inp.useCompression:
    compress_vtu(foutfile, method=compress_method, verbose=inp.verbose)
  if inp.progress:
    if bart is not None:
      bart.next()

def main_cli(args=None) -> None:
  ################################################################################################
  # Get the commandline arguments
  args = parser.parse_args(args=args)
  # if args.variablenames == []:sys.exit(">>>ERROR: no variable names are supplied to be processed!!!")
  ################################################################################################
  # The main Script
  print_header()
  print_input_info(args)
  inp = get_inputs(args)
  check_variables(inp)
  ################################################################################################
  # read topology and get number of elements, number of nodes per elements
  tp  = LoadTopology(inp)
  export_boundary(tp, inp)
  ################################################################################################
  # now convert all requested files, looping over all requested times
  lastvariable = [None for _ in inp.vars]
  lastspace    = None
  lastdeform   = None
  bart         = None
  if inp.cores is None:
    if args.subauto:
      if inp.progress: bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di)
      for time in range(inp.i0, inp.it, inp.di):
        jlist = glob.glob('145592*.jpg')
        for j in jlist:
          result = re.search('-(.*).D', j)
          export_data_iter(tp, inp, result.group(1), lastvariable, lastspace, lastdeform, bart)
    elif args.subiter is not None:
      if inp.progress: bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di * (args.subiter[1] - args.subiter[0] + 1)//args.subiter[2])
      for time in range(inp.i0, inp.it, inp.di):
        export_data_iter(tp, inp, time, lastvariable, lastspace, lastdeform,  bart)
        for j in range(args.subiter[0], args.subiter[1] + 1, args.subiter[2]):
          export_data_iter(tp, inp, str(time)+'.'+str(j), lastvariable, lastspace, lastdeform)
    else:
      if inp.progress:
        bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di)
      for time in range(inp.i0, inp.it, inp.di):
        export_data_iter(tp, inp, time, lastvariable, lastspace, lastdeform,  bart)
  else:
    with futures.ProcessPoolExecutor(inp.cores) as exec:
      future_jobs=[]
      if args.subauto:
        if inp.progress: bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di)
        for time in range(inp.i0, inp.it, inp.di):
          future_jobs.append(exec.submit(export_data_iter, tp, inp, time, lastvariable, lastspace, lastdeform))
          jlist = glob.glob(f'*-{time}.*.D')
          for j in jlist:
            result = re.search('-(.*).D', j)
            future_jobs.append(exec.submit(export_data_iter, tp, inp, str(j), lastvariable, lastspace, lastdeform))
      elif args.subiter is not None:
        if inp.progress: bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di * (args.subiter[1] - args.subiter[0] + 1)//args.subiter[2])
        for time in range(inp.i0, inp.it, inp.di):
          future_jobs.append(exec.submit(export_data_iter, tp, inp, time, lastvariable, lastspace, lastdeform))
          for j in range(args.subiter[0], args.subiter[1] + 1, args.subiter[2]):
            future_jobs.append(exec.submit(export_data_iter, tp, inp, str(time)+'.'+str(j), lastvariable, lastspace, lastdeform))
      else:
        if inp.progress: bart = progress_bar('Processing', max=(inp.it - inp.i0)//inp.di)
        for time in range(inp.i0, inp.it, inp.di):
          future_jobs.append(exec.submit(export_data_iter, tp, inp, time, lastvariable, lastspace, lastdeform))
      for _ in futures.as_completed(future_jobs):
        bart.next()


    # write header
    # if progress: bart.finish()
  print("################################################################################################")
  if args.time_series is not None:
    from .make_vtu_series import check_args, print_cmd_header, xml_write_header, xml_write_content, xml_write_footer
    inp = check_args(args)
    print_cmd_header(inp)
    bar = progress_bar('Processing', max=inp.nt)
    with open(os.path.join(inp.folder, inp.prefix+".pvd"), "w") as f:
      xml_write_header(f)
      for i in range(inp.i0, inp.it, inp.di):
        xml_write_content(f, os.path.join(inp.folder, f"{inp.prefix}-{i}.vtu"), inp.time[i])
        bar.next()
      xml_write_footer(f)
    bar.finish()
    print("    The process is complete!")
    print("################################################################################################")

if __name__ == "__main__":
  main_cli()

