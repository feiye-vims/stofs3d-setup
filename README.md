# stofs3d-setup

Scripts for setting up **STOFS-3D Atlantic** and related compound flooding modeling applications based on SCHISM.

These scripts automate common preprocessing tasks such as mesh preparation, forcing setup, and workflow configuration.

---

## Dependencies

The following Python packages are required:

- `fiona`
- `gsw`
- `pyshp` *(temporary; will be removed in a future update)*

---

## Installation

### 1. Install `pyschism` (development version)

Install the latest development version from GitHub:

```bash
git clone https://github.com/schism-dev/pyschism.git
cd pyschism
pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag is currently required due to packaging issues in the upstream repository.

---

### 2. Install `pylibs-ocean`

Only the core functionality is required:

```bash
pip install pylibs-ocean
```

---

### 3. Install experimental pylibs utilities

```bash
pip install git+https://github.com/wzhengui/pylibs.git#subdirectory=pylib_experimental
```

---

### 4. Temporary dependency

This dependency will be removed in a future update:

```bash
pip install git+https://github.com/feiye-vims/schism_py_pre_post.git
```

---

## Notes

- This repository assumes familiarity with the **SCHISM/STOFS modeling workflow**.
- Several dependencies are currently transitional and will be simplified as upstream packages stabilize.


