# cheartpy

Welcome to the documentation for **cheartpy**, a multi-package Python namespace monorepo designed for advanced computational mechanics workflows.

---

## 📦 Monorepo Workspace Packages

This project is managed as a unified workspace. Below is an overview of the core libraries and plugins available:

- **[Core IO (cheartpy.io)](reference/cheartpy/io/index.html)**: Core input/output operations, configuration parsers, and workspace discovery.
- **[Abaqus Plugin (cheartpy-abaqus)](reference/cheartpy/abaqus/index.html)**: Integration utilities, meshing translations, and scripting hooks for ABAQUS.
- **[Finite Element Solver (cheartpy-fe)](reference/cheartpy/fe/index.html)**: Core finite element parsing arrays, command tools, and time-stepping loops.
- **[GMSH Interface (cheartpy-gmsh)](reference/cheartpy/gmsh/index.html)**: Native bindings and geometric pipeline controls for GMSH geometry tracking.
- **[Mesh Engine (cheartpy-mesh)](reference/cheartpy/mesh/index.html)**: Advanced internal meshing engines and structural layout tools.
- **[Paraview (cheartpy-paraview)](reference/cheartpy/paraview/index.html)**: State file generators and visualization pipelines.
- **[VTK Utilities (cheartpy-vtk)](reference/cheartpy/vtk/index.html)**: Direct VTK data wrappers and XML element interfaces.

---

## 🚀 Quick Start

### Installation

Because this project utilizes a modern workspace environment, you can install the complete stack or individual plugins seamlessly using `uv`:

```bash
# Install the full workspace development environment
uv sync

# Run a script utilizing the core library
uv run python my_script.py
```

### Basic Usage

To import components from across the namespace packages into your workflows:

```python
import cheartpy.io as cio
import cheartpy.mesh as cmesh

# Load structural parameters
mesh_data = cmesh.load_mesh("cell_lin.inp")
```

---

## 🛠️ Development & Contributions

If you are adding functionality to a specific subpackage (e.g., `cheartpy-mesh`), make sure to work within its dedicated subdirectory:

1.  Code logic lives inside `cheartpy-*/src/cheartpy/`
2.  Tests are located in the root `tests/` directory
3.  Documentation docstrings are parsed **automatically** on every build line.
