# type: ignore
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:49:36 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io

# TODO Add verbose option to disable warnings
verbose = True


vol_to_surf_elem = {'hexahedron': 'quad',
                    'hexahedron27': 'quad9',
                    'triangle': 'line',
                    'tetra': 'triangle',
                    'tetra10': 'triangle6'}


# load CHeart files
def read_mesh(path, meshio=False, element=None, tfile=None, xfile=None):
    # Load mesh
    if xfile is None:
        xyz = np.loadtxt(path + '_FE.X', skiprows = 1)
    else:
        xyz = np.loadtxt(path + xfile, skiprows = 1)
    if tfile is None:
        ien = np.loadtxt(path + '_FE.T', skiprows = 1, dtype=int) - 1
    else:
        ien = np.loadtxt(path + tfile, skiprows = 1, dtype=int) - 1
    try: bfile = np.loadtxt(path + '_FE.B', skiprows = 1)
    except: bfile = np.array([])

    ien, element = get_element_type(ien, element=element, bfile=bfile)

    if meshio:
        return io.Mesh(xyz, {element: ien})
    else:
        return xyz, ien, element


def read_bfile(path, element=None):
    array = np.loadtxt(path + '_FE.B', skiprows = 1, dtype=int)
    array[:,0:-1] = array[:,0:-1] - 1
    if element is None:
        _, _, element = read_mesh(path)
    array[:,1:-1] = connectivity_CH2vtu(vol_to_surf_elem[element], array[:,1:-1])    # Correct order to vtu order
    return array


def read_fibers(path, append2d=False):
    fibers = np.loadtxt(path, skiprows = 1)

    if fibers.shape[1] == 9:
        f = fibers[:,0:3]
        s = fibers[:,3:6]
        n = fibers[:,6:9]
        return f, s, n
    elif fibers.shape[1] == 4:
        f = fibers[:,0:2]
        s = fibers[:,2:4]
        if append2d:
            f = np.vstack([f.T, np.zeros(f.shape[0])]).T
            s = np.vstack([s.T, np.zeros(s.shape[0])]).T
        return f, s


def read_dfile(path, **kwargs):
    array = np.loadtxt(path, skiprows = 1, ndmin=1, **kwargs)

    return array


def read_scalar_dfiles(path, times, return_incomplete=False):
    st, et, inc = times
    et += 1
    t = np.arange(st, et, inc)
    nts = len(t)

    array = np.zeros(nts)
    incomplete = False
    for i, cont in enumerate(range(st, et, inc)):
        try:
            array[i] = read_dfile(path + '-%i' % cont + '.D')
        except:
            lts = cont
            incomplete = True
            break

    array = np.array(array)
    if return_incomplete:
        return array
    else:
        if not incomplete:
            return array
        else:
            raise Exception('Missing file: ' +  path + '-%i' % lts + '.D')


def get_element_type(ien, element=None, ch2vtu=True, bfile=np.array([])):
    if element == None:
        element = get_element_type_by_nnodes(ien, bfile)

    if ch2vtu:
        ien = connectivity_CH2vtu(element, ien)
    else:
        ien = connectivity_vtu2CH(element, ien)

    return ien, element


def connectivity_CH2vtu(element, ien):
    # CH to vtu node numeration
    if element == 'line3':
        ien = ien[:, np.array([0, 2, 1])]
    elif element == 'triangle6':
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif element == 'quad':
        ien = ien[:, np.array([0, 1, 3, 2])]
    elif element == 'quad9':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 7, 8, 5,
                               6])]
        # ien = ien[:, np.array([0, 2, 8, 6,
        #                        1, 5, 7, 3,
        #                        4])]
    elif element == 'hexahedron':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6])]

    elif element == 'tetra10':
        ien = ien[:, np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])]
    elif element == 'hexahedron27':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6,
                               8, 11, 12, 9,
                               22, 25, 26, 23,
                               13, 15, 21, 19,
                               16, 18, 14, 20,
                               10, 24, 17])]

    return ien

def face_array(element):
    # This returns the array that return the faces of an element
    if element == 'triangle':
        array = np.array([[0,1],[1,2],[2,0]])
    elif element == 'tetra':
        array = np.array([[0,1,2],[1,2,3],[0,2,3],[0,1,3]])
    else:
        raise 'Not Implemented'
    return array




def get_element_type_by_nnodes(ien, bfile=np.array([])):
    if len(ien.shape) == 1:
        return 'point'
    if ien.shape[1] == 2:
        element = 'line'
    elif ien.shape[1] == 3:
        if bfile.size == 0:
            element = 'triangle'
            if verbose:
                print('WARNING: No .Bfile, ambiguous number of nodes, choosing triangle')
        else:
            if bfile.shape[1] == 3:
                element = 'line3'
            elif bfile.shape[1] == 4:
                element = 'triangle'
            else:
                if verbose:
                    print('WARNING: Something wrong with .Bfile, ambiguous number of nodes, choosing triangle')
                element = 'triangle'
    elif ien.shape[1] == 6:
        element = 'triangle6'
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif ien.shape[1] == 4:     # TODO how do I handle this?
        if bfile.size == 0:
            element = 'tetra'
            if verbose:
                print('WARNING: No .Bfile, ambiguous number of nodes, choosing tetra')
        else:
            if bfile.shape[1] == 4:
                element = 'quad'
            elif bfile.shape[1] == 5:
                element = 'tetra'
            else:
                element = 'tetra'
                if verbose:
                    print('WARNING: Something wrong with .Bfile, ambiguous number of nodes, choosing tetra')
    elif ien.shape[1] == 9:
        element = 'quad9'
    elif ien.shape[1] == 8:
        element = 'hexahedron'
    elif ien.shape[1] == 10:
        element = 'tetra10'
    elif ien.shape[1] == 27:
        element = 'hexahedron27'
    else:
        print('WARNING: element not found')
    return element

# save CHeart files


def connectivity_vtu2CH(element, ien):  # TODO
    # CH to vtu node numeration
    if element == 'line3':
        ien = ien[:, np.array([0, 2, 1])]
    elif element == 'triangle6':
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif element == 'quad':
        ien = ien[:, np.array([0, 1, 3, 2])]
    elif element == 'quad9':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 7, 8, 5,
                               6])]
    elif element == 'tetra10':
        ien = ien[:, np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])]
    elif element == 'hexahedron':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6])]

    elif element == 'hexahedron27':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6,
                               8, 11, 24, 9,
                               10, 16, 22, 17,
                               20, 26, 21, 19,
                               23, 18, 12, 15,
                               25, 13, 14])]

    return ien


def write_xfile(fname, pts):    # TODO check if the extension is correct, if not add it
    np.savetxt(fname, pts, fmt='%30.15f',
               delimiter='\t',  header = str(pts.shape[0]) + '\t' + str(pts.shape[1]),
               comments = '')

def write_tfile(fname, elems, pts):
    np.savetxt(fname, elems+1, fmt='%i',
           delimiter='\t', header = str(elems.shape[0]) + '\t' + str(pts.shape[0]),
           comments = '')

def write_mesh(fname, pts, elems, element=None):
    write_xfile(fname + '_FE.X', pts)
    elems, element = get_element_type(elems, ch2vtu=False, element=element)

    write_tfile(fname + '_FE.T', elems, pts)

def write_bfile(fname, bound):
    boundary = bound.copy()
    # Fix the numeration of elements and points (or unfix it depending who you ask)
    boundary[:,0:-1] += 1

    bfaces = boundary[:,1:-1]
    bfaces, _ = get_element_type(bfaces, ch2vtu=False)
    boundary[:,1:-1] = bfaces
    np.savetxt(fname+'_FE.B', boundary, fmt='%i',
           delimiter='\t', header = str(boundary.shape[0]), comments = '')


def write_dfile(fname, array, fmt=None):     # TODO check if name finish in D or not
    shape = array.shape
    size = array.size
    if size == 1:
        np.savetxt(fname, array, header = str(1) + '\t' + str(1),
               comments = '')
    else:
        if len(shape) == 1:
            s1 = shape[0]
            s2 = 1
        else:
            s1 = shape[0]
            s2 = shape[1]
        if fmt == None:
            fmt = '%30.15f'
        np.savetxt(fname, array, fmt=fmt,header = str(s1) + '\t' + str(s2),
               comments = '')

def write_specific(fname, nodes, values, fmt=None):
    if len(values.shape) == 1:
        specific = np.vstack([nodes+1, values]).T
        fmt = ['%i', '%f']
    else:
        specific = np.hstack([nodes[:,None]+1, values])
        fmt = ['%i'] + ['%f']*values.shape[1]
    np.savetxt(fname, specific, header = str(len(specific)), comments = '', fmt = fmt)

def read_specific(fname):
    data = np.loadtxt(fname, skiprows=1)
    nodes = data[:,0].astype(int) - 1
    values = data[:,1:]
    return nodes, values

def vtu_to_mesh(iname, oname, dim=3):
    mesh = io.read(iname)
    pts = mesh.points
    if len(mesh.cells) > 1:
        elem_nodes = []
        for c in mesh.cells:
            elem_nodes = c.data.shape[1]
        ind = np.argmax(elem_nodes)
        print('WARNING: Multiple types of cells defined. Choosing ' + c.type)
        elem = mesh.cells[ind].type
        elems = mesh.cells[ind].data

    else:
        elem = mesh.cells[0].type
        elems = mesh.cells[0].data

    if dim == 2:
        pts = pts[:,0:2]

    # This is just to fix the node order in case is needed
    ien, elem = get_element_type(elems, element=elem)

    write_xfile(oname + '_FE.X', pts)
    write_tfile(oname + '_FE.T', ien, pts)


# to .vtu
def mesh_to_vtu(mesh_path, out_name, elem=None, xfile=None):

    X, T, elem = read_mesh(mesh_path, element=elem)
    if xfile is not None:
        X = read_dfile(xfile)

    io.write_points_cells(out_name, X, {elem: T})

def bfile_to_vtu(mesh_path, out_name, element = None):
    X, T, _ = read_mesh(mesh_path)
    B = read_bfile(mesh_path)
    ien = B[:,1:-1]
    marker = B[:,-1]
    element = get_element_type_by_nnodes(ien, B)
    io.write_points_cells(out_name, X, {element: ien},
                          cell_data = {'patches': [marker]})

def bfile_to_blockmesh(mesh_path):
    X, T, _ = read_mesh(mesh_path)
    B = read_bfile(mesh_path)
    ien = B[:,1:-1]
    marker = B[:,-1]
    ien, element = get_element_type(ien)

    nmarkers = np.unique(marker)
    cells = []
    for i in nmarkers:
        cells.append(io.CellBlock(element, ien[marker==i]))

    return io.Mesh(X, cells)


def dfile_to_vtu(D_path, out_name, mesh_path=None, mesh=None, var_name = 'f',
                 array_type = 'points', element=None, inverse=False, xfile=None):

    if not mesh_path == None:
        mesh = read_mesh(mesh_path, element=element, meshio=True)
    if xfile is not None:
        mesh.points = read_dfile(xfile)
    X = mesh.points

    # Check if 2D data
    if X.shape[1] == 2:
        X = np.hstack([X, np.zeros([X.shape[0],1])])

    if type(D_path) == list:
        for d, v in zip(D_path, var_name):
            array = np.loadtxt(d, skiprows = 1)
            if len(array.shape) != 1:
                if array.shape[1] == 2:
                    array = np.hstack([array, np.zeros([array.shape[0],1])])

            if array_type == 'points':
                # Check that array has correct size
                assert len(array) == len(mesh.points), 'number of data points is not the same as the number of mesh points'
                mesh.point_data[v] = array
            elif array_type == 'cells':
                # Check that array has correct size
                assert len(array) == len(mesh.cells[0].data), 'number of data points is not the same as the number of elements'
                mesh.cell_data[v] = [array]


    else:
        array = np.loadtxt(D_path, sk...