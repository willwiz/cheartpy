# cheartpy

Welcome to the documentation for **cheartpy**, a multi-package Python namespace monorepo designed for working with Cheart.

---

## 📦 Monorepo Workspace Packages

This project is managed as a unified workspace. Below is an overview of the core libraries and plugins available:

- **[Core IO (cheartpy.io)](reference/cheartpy/io/index.html)**: Core input/output operations, configuration parsers, and workspace discovery.
- **[Abaqus Mesh Interface (cheartpy-abaqus)](reference/cheartpy/abaqus/index.html)**: Importing and exporting to abaqus meshes
- **[Finite Element Python Interface (cheartpy-fe)](reference/cheartpy/fe/index.html)**: API for generating Cheart PFiles.
- **[GMSH Interface (cheartpy-gmsh)](reference/cheartpy/gmsh/index.html)**: Native bindings and api function for importing, optimizing, and exporting meshes to Gmsh.
- **[Cheart Mesh Interface (cheartpy-mesh)](reference/cheartpy/mesh/index.html)**: Creating and working with meshes in Cheart format.
- **[Paraview Interface (cheartpy-paraview)](reference/cheartpy/paraview/index.html)**: Generate and working with VTU files for paraview outputs.
- **[VTK and Element Utilities (cheartpy-vtk)](reference/cheartpy/vtk/index.html)**: Utilities for converting between various element types and VTK data structures.

---

## 🚀 Quick Start

### Installation

Because this project utilizes a modern workspace environment, you can install the complete stack or individual plugins seamlessly using `uv`:

```bash
# Install the full workspace development environment
uv add "cheartpy[...] @ git+https://github.com/willwiz/cheartpy.git"
uv sync
```

## Installing scripts as global tools

### Tool `uv`

To be able to run programs globally while not interferring with the python environment

```bash
uv tool install "cheartpy[paraview] @ git+https://github.com/willwiz/cheartpy.git" --with-executables-from cheartpy-paraview
```

### Tool `pipx`

```bash
pipx install "cheartpy[paraview] @ git+https://github.com/willwiz/cheartpy.git" --include-deps
```

## Development Mode

```bash
git clone https://github.com/willwiz/cheartpy.git
cd cheartpy
uv sync
```

## Making the documentations

Install in Development Mode and run zensical from with in the root directory:

```bash
cd cheartpy
uv run zensical build --clean
```

## 🛠️ Development & Contributions

Work in progress! Maybe submit a pull request.
