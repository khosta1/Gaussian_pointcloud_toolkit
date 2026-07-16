# Gaussian Pointcloud Toolkit

A small Python/tkinter tool to **clean COLMAP `points3D.txt` sparse point clouds**
before training a Gaussian Splatting model (e.g. RealityCapture → clean →
Lichtfeld Studio / nerfstudio / any 3DGS trainer).

It removes outlier / floating points with **Statistical Outlier Removal (SOR)**
in an **iterative 3D editor** — no CloudCompare round-trip, no export/reimport.

![mode](https://img.shields.io/badge/python-3.9%2B-blue) ![gui](https://img.shields.io/badge/gui-tkinter-green)

## Features

- **Landing page = the SOR cleaner.** Open a `points3D.txt` and work directly.
- **Iterate loop:** *Preview* (marks outliers red) → *Delete red + reframe* →
  repeat on the already-cleaned cloud, with full **Undo** history.
- **3D viewer draws every point** via a hand-rolled software renderer
  (project → pixel buffer → PNG → `tk.PhotoImage`). Drag to rotate, wheel to zoom.
  Grey = kept, red = will-delete. Live counts (Original / Current / Marked / passes).
- **Fast SOR:** uses a SciPy KD-tree when available (~0.1 s for 200k points),
  with a pure-python spatial-grid fallback if SciPy isn't installed.
- **Toolkit menu** (home for future helper scripts): `points3D.txt ↔ PLY`
  converters that correctly handle COLMAP's variable-length track columns.
- **Headless CLI** for batch use.

## Install

```bash
pip install numpy scipy   # optional but recommended (fast SOR); GUI needs neither
```

Standard-library `tkinter` provides the GUI. No other dependencies.

## Run

```bash
python Gaussian_pointcloud_toolkit.py
```

or double-click `Gaussian_pointcloud_toolkit.bat` on Windows.

**Workflow:** *File → Open* your `points3D.txt` → set **Neighbours (k)** and
**Std-dev (sigma)** → **Preview SOR** → **Delete red + reframe** → repeat →
**Save**. The saved file replaces `points3D.txt` next to your untouched
`cameras.txt` / `images.txt`.

### Headless / batch

```bash
python Gaussian_pointcloud_toolkit.py --sor in.txt out.txt --neighbors 6 --sigma 1.0
```

## Tips

- SOR removes points whose mean distance to their *k* nearest neighbours exceeds
  `mean + sigma·std`. **Lower sigma (0.8)** = more aggressive; **higher (1.5–2.0)**
  = gentle.
- SOR is statistical, so it always marks *something* — watch the red preview and
  stop when it starts eating real surface detail.
- **Only delete points — never move or scale the cloud.** That keeps it aligned
  with the COLMAP cameras.

## License

MIT
