## Installing as a package in virtual environment

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
