"""
Gaussian Pointcloud Toolkit  (COLMAP SOR cleaner + format converters)
=====================================================================

Landing page = the iterative SOR cleaner: open a points3D.txt, then loop
Preview -> Delete -> reframe to strip outliers/floaters in-session, with a live
3D viewer that draws EVERY point (kept = grey, removed = red) and shows counts.

  File menu    : Open / Save points3D.txt
  Toolkit menu : points3D.txt <-> PLY converters (home for future helper tools)

Headless / batch use (no window):
  python Gaussian_pointcloud_toolkit.py --sor in.txt out.txt --neighbors 6 --sigma 1.0

SOR uses a scipy KD-tree when installed (~0.1s for 200k points); if scipy is
missing it automatically falls back to a slower pure-python implementation.
GUI is plain tkinter. No CloudCompare needed.
    pip install numpy scipy   # recommended, for fast SOR
"""

import os
import sys
import math
import zlib
import struct
import base64
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _png_base64(W, H, buf):
    """Encode an RGB byte buffer (len W*H*3) as a base64 PNG string for Tk."""
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))
    raw = bytearray()
    row = W * 3
    for y in range(H):
        raw.append(0)                 # filter byte 0 (none)
        raw += buf[y * row:(y + 1) * row]
    comp = zlib.compress(bytes(raw), 1)
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)   # 8-bit RGB
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"IDAT", comp) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode("ascii")


# ==========================================================================
#  I/O  (COLMAP points3D.txt  and  ascii PLY)
# ==========================================================================
def load_points3d(path):
    """Return list of (xs, ys, zs, r, g, b) as strings (originals preserved)."""
    pts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if len(c) < 7:
                continue
            # cols: ID X Y Z R G B ERROR track...
            pts.append((c[1], c[2], c[3], c[4], c[5], c[6]))
    return pts


def save_points3d(path, pts):
    """Write kept points back to COLMAP format (ERROR=0, empty track)."""
    with open(path, "w") as f:
        f.write("# 3D point list\n# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        f.write(f"# Number of points: {len(pts)}\n")
        for i, (x, y, z, r, g, b) in enumerate(pts, start=1):
            f.write(f"{i} {x} {y} {z} {r} {g} {b} 0\n")


def points3d_to_ply(src, dst, log):
    pts = load_points3d(src)
    with open(dst, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in pts:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    log(f"OK  -  wrote {len(pts)} points to:\n     {dst}")


def ply_to_points3d(src, dst, log):
    with open(src, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    if not lines or lines[0].strip().lower() != "ply":
        raise ValueError("Not a PLY file (missing 'ply' header).")
    if "format ascii" not in lines[1].lower():
        raise ValueError("PLY is not ASCII. Re-save it from CloudCompare as "
                         "PLY with 'ASCII' encoding.")
    props, count, hdr_end = [], 0, 0
    for i, ln in enumerate(lines):
        t = ln.split()
        if t[:1] == ["element"] and len(t) >= 3 and t[1] == "vertex":
            count = int(t[2])
        elif t[:1] == ["property"]:
            props.append(t[-1])
        elif ln.strip() == "end_header":
            hdr_end = i + 1
            break

    def idx(*names):
        for n in names:
            if n in props:
                return props.index(n)
        return None

    ix, iy, iz = idx("x"), idx("y"), idx("z")
    ir, ig, ib = idx("red", "r"), idx("green", "g"), idx("blue", "b")
    if None in (ix, iy, iz):
        raise ValueError("Could not find x/y/z in the PLY header.")

    pts = []
    for ln in lines[hdr_end:hdr_end + count]:
        c = ln.split()
        if len(c) <= max(ix, iy, iz):
            continue
        x, y, z = c[ix], c[iy], c[iz]
        if ir is not None and len(c) > max(ir, ig, ib):
            r, g, b = c[ir], c[ig], c[ib]
            if "." in r:  # float 0..1 -> 0..255
                r, g, b = (str(int(round(float(v) * 255))) for v in (r, g, b))
        else:
            r, g, b = "128", "128", "128"
        pts.append((x, y, z, r, g, b))
    save_points3d(dst, pts)
    log(f"OK  -  wrote {len(pts)} points to:\n     {dst}")


# ==========================================================================
#  SOR engine  (pure-python spatial grid, no dependencies)
# ==========================================================================
def compute_sor(fx, fy, fz, k=6, sigma=1.0, progress=None):
    """
    Statistical Outlier Removal.
    For every point: mean distance to its k nearest neighbours.
    Remove points whose mean distance > (global_mean + sigma * global_std).

    Uses a scipy KD-tree when available (fast, ~1s for 200k points), and
    falls back to a pure-python spatial grid otherwise.

    Returns (keep_mask, mean_dist, threshold, mean, std).
    """
    n = len(fx)
    if n == 0:
        return [], [], 0.0, 0.0, 0.0
    if _HAVE_SCIPY:
        return _sor_scipy(fx, fy, fz, k, sigma)
    return _sor_python(fx, fy, fz, k, sigma, progress)


def _sor_scipy(fx, fy, fz, k, sigma):
    pts = np.empty((len(fx), 3), dtype=np.float64)
    pts[:, 0] = fx
    pts[:, 1] = fy
    pts[:, 2] = fz
    tree = cKDTree(pts)
    kq = min(k + 1, len(pts))                 # +1 because query returns self
    dist, _ = tree.query(pts, k=kq, workers=-1)
    if kq > 1:
        mean_dist = dist[:, 1:].mean(axis=1)  # drop the self (distance 0)
    else:
        mean_dist = np.zeros(len(pts))
    mean = float(mean_dist.mean())
    std = float(mean_dist.std())
    threshold = mean + sigma * std
    keep = (mean_dist <= threshold).tolist()
    return keep, mean_dist.tolist(), threshold, mean, std


def _sor_python(fx, fy, fz, k=6, sigma=1.0, progress=None):
    n = len(fx)
    minx, maxx = min(fx), max(fx)
    miny, maxy = min(fy), max(fy)
    minz, maxz = min(fz), max(fz)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz

    # cell size ~ a couple of points per cell on average (radius-1 search
    # then usually yields enough neighbours in a single pass)
    vol = dx * dy * dz
    if vol > 0:
        cell = 1.4 * (vol / n) ** (1.0 / 3.0)
    else:
        span = max(dx, dy, dz)
        cell = span / max(n ** (1.0 / 3.0), 1.0)
    if cell <= 0:
        cell = 1e-6
    inv = 1.0 / cell

    # integer-keyed grid on a padded lattice -> neighbour keys are just
    # base_key + offset, with no tuple creation and no bounds checks.
    nb = int(dy * inv) + 3          # padded axis sizes (+1 each side, +1 slack)
    nc = int(dz * inv) + 3
    NBNC = nb * nc

    key = [0] * n
    grid = {}
    grid_get = grid.get
    for i in range(n):
        a = int((fx[i] - minx) * inv) + 1
        b = int((fy[i] - miny) * inv) + 1
        c = int((fz[i] - minz) * inv) + 1
        ki = (a * nb + b) * nc + c
        key[i] = ki
        bucket = grid_get(ki)
        if bucket is None:
            grid[ki] = [i]
        else:
            bucket.append(i)

    # 27 neighbour offsets in key space (precomputed once)
    offsets = [(di * nb + dj) * nc + dk
               for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)]

    mean_dist = [0.0] * n
    big = math.sqrt((dx * dx + dy * dy + dz * dz) or 1.0)
    sqrt = math.sqrt

    for i in range(n):
        xi, yi, zi = fx[i], fy[i], fz[i]
        ki = key[i]
        dists = []
        add = dists.append
        for off in offsets:
            bucket = grid_get(ki + off)
            if bucket is None:
                continue
            for j in bucket:
                if j == i:
                    continue
                ddx = fx[j] - xi
                ddy = fy[j] - yi
                ddz = fz[j] - zi
                add(ddx * ddx + ddy * ddy + ddz * ddz)

        if len(dists) >= k:
            dists.sort()
            s = 0.0
            for d2 in dists[:k]:
                s += sqrt(d2)
            mean_dist[i] = s / k
        elif dists:
            dists.sort()
            s = 0.0
            for d2 in dists:
                s += sqrt(d2)
            mean_dist[i] = s / len(dists)
        else:
            mean_dist[i] = big          # isolated point -> outlier

        if progress and (i & 0x3FFF) == 0:
            progress(i, n)

    # global statistics
    mean = sum(mean_dist) / n
    var = 0.0
    for d in mean_dist:
        var += (d - mean) * (d - mean)
    std = math.sqrt(var / n)
    threshold = mean + sigma * std

    keep = [d <= threshold for d in mean_dist]
    return keep, mean_dist, threshold, mean, std


# ==========================================================================
#  3D viewer  (software renderer -> draws every point)
# ==========================================================================
class App(tk.Tk):
    """Main window: the iterative SOR cleaner. Converters live in the menu."""
    W, H = 860, 600  # render size in pixels

    def __init__(self):
        super().__init__()
        self.title("COLMAP point cleaner  -  SOR")

        # working set (empty until a points3D.txt is opened)
        self.pts = []
        self.fx, self.fy, self.fz = [], [], []
        self.keep = []
        self.idx_keep, self.idx_drop = [], []
        self.default_out = os.path.join(os.path.dirname(__file__),
                                        "points3D_clean.txt")
        self.orig_total = 0
        self.applied = 0                 # number of committed delete passes
        self.history = []                # snapshots for undo

        self.zoom = 1.0
        self.ay = 0.6   # yaw
        self.ax = 0.35  # pitch
        self._last = None
        self._pending = False
        self.cx = self.cy = self.cz = 0.0
        self.base_scale = 1.0
        self.k_var = tk.IntVar(value=6)
        self.s_var = tk.DoubleVar(value=1.0)

        self._build_menu()

        # ---- controls (left panel) ----
        panel = ttk.Frame(self, padding=8)
        panel.pack(side="left", fill="y")

        ttk.Button(panel, text="Open points3D.txt ...",
                   command=self.open_file).pack(fill="x", pady=(0, 8))

        ttk.Label(panel, text="SOR parameters",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        pf = ttk.Frame(panel)
        pf.pack(anchor="w", pady=4)
        ttk.Label(pf, text="Neighbours (k)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(pf, from_=2, to=50, width=6,
                    textvariable=self.k_var).grid(row=0, column=1, padx=4)
        ttk.Label(pf, text="Std-dev (sigma)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(pf, from_=0.1, to=5.0, increment=0.1, width=6,
                    textvariable=self.s_var).grid(row=1, column=1, padx=4)

        ttk.Button(panel, text="1.  Preview SOR  (mark red)",
                   command=self.preview).pack(fill="x", pady=(8, 2))
        ttk.Button(panel, text="2.  Delete red  +  reframe",
                   command=self.apply_delete).pack(fill="x", pady=2)
        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Button(panel, text="Reframe camera",
                   command=self.reframe).pack(fill="x", pady=2)
        ttk.Button(panel, text="Undo last delete",
                   command=self.undo).pack(fill="x", pady=2)

        self.count_lbl = tk.Label(panel, justify="left", anchor="w",
                                  font=("Consolas", 9))
        self.count_lbl.pack(anchor="w", fill="x", pady=(10, 0))

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Button(panel, text="Save points3D.txt",
                   command=self.save).pack(fill="x")
        tk.Label(panel, justify="left", fg="#666", font=("Segoe UI", 8),
                 text="\nLoop: Preview -> Delete -> repeat.\n"
                      "Delete auto-reframes on\nwhat is left.\n\n"
                      "Drag = rotate   Wheel = zoom\n"
                      "Grey = kept     Red = will delete"
                 ).pack(anchor="w")

        # ---- canvas ----
        self.canvas = tk.Canvas(self, width=self.W, height=self.H,
                                bg="#0f0f14", highlightthickness=0)
        self.canvas.pack(side="left")
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)

        self._split_indices()
        self.update_counts()
        self.render()

    # -- fit camera (centre + scale) to the current working set ---------
    def _fit(self):
        fx, fy, fz = self.fx, self.fy, self.fz
        if not fx:
            self.cx = self.cy = self.cz = 0.0
            self.base_scale = 1.0
            return
        self.cx = (min(fx) + max(fx)) / 2
        self.cy = (min(fy) + max(fy)) / 2
        self.cz = (min(fz) + max(fz)) / 2
        ext = max(max(fx) - min(fx),
                  max(fy) - min(fy),
                  max(fz) - min(fz)) or 1.0
        self.base_scale = 0.45 * min(self.W, self.H) / ext

    # -- keep/remove index lists (drawing order: kept first, red on top) --
    def _split_indices(self):
        self.idx_keep = [i for i, k in enumerate(self.keep) if k]
        self.idx_drop = [i for i, k in enumerate(self.keep) if not k]

    def update_counts(self):
        cur = len(self.pts)
        marked = len(self.idx_drop)
        pct = (100.0 * marked / cur) if cur else 0.0
        self.count_lbl.config(
            text=f"Original : {self.orig_total:>8}\n"
                 f"Current  : {cur:>8}\n"
                 f"Marked   : {marked:>8}  ({pct:4.1f}%)\n"
                 f"Deletes  : {self.applied:>8} pass(es)")

    # ---- rendering ----------------------------------------------------
    def render(self):
        W, H = self.W, self.H
        if not self.pts:
            self.canvas.itemconfig(self.img_id, image="")
            self.canvas.delete("hint")
            self.canvas.create_text(
                W // 2, H // 2, tags="hint", fill="#666",
                font=("Segoe UI", 13), justify="center",
                text="Open a points3D.txt to begin\n"
                     "( File > Open,  or the button on the left )")
            return
        self.canvas.delete("hint")
        buf = bytearray(b"\x0f\x0f\x14" * (W * H))  # background fill

        cosy, siny = math.cos(self.ay), math.sin(self.ay)
        cosx, sinx = math.cos(self.ax), math.sin(self.ax)
        scale = self.base_scale * self.zoom
        ox, oy = W // 2, H // 2
        fx, fy, fz = self.fx, self.fy, self.fz
        cx, cy, cz = self.cx, self.cy, self.cz

        def paint(indices, r, g, b):
            for i in indices:
                X = fx[i] - cx
                Y = fy[i] - cy
                Z = fz[i] - cz
                x1 = X * cosy + Z * siny
                z1 = -X * siny + Z * cosy
                y2 = Y * cosx - z1 * sinx
                px = int(ox + x1 * scale)
                py = int(oy - y2 * scale)
                if 0 <= px < W and 0 <= py < H:
                    o = (py * W + px) * 3
                    buf[o] = r
                    buf[o + 1] = g
                    buf[o + 2] = b

        paint(self.idx_keep, 140, 150, 170)   # grey-blue kept
        paint(self.idx_drop, 255, 45, 45)      # red removed

        self.photo = tk.PhotoImage(master=self.canvas,
                                   data=_png_base64(W, H, buf), format="png")
        self.canvas.itemconfig(self.img_id, image=self.photo)

    def _schedule(self):
        if not self._pending:
            self._pending = True
            self.after_idle(self._do_render)

    def _do_render(self):
        self._pending = False
        self.render()

    # ---- interaction --------------------------------------------------
    def _press(self, e):
        self._last = (e.x, e.y)

    def _drag(self, e):
        if self._last is None:
            self._last = (e.x, e.y)
            return
        dx = e.x - self._last[0]
        dy = e.y - self._last[1]
        self._last = (e.x, e.y)
        self.ay += dx * 0.01
        self.ax += dy * 0.01
        self.ax = max(-1.5, min(1.5, self.ax))
        self._schedule()

    def _wheel(self, e):
        self.zoom *= 1.1 if e.delta > 0 else (1 / 1.1)
        self._schedule()

    # ---- iterate loop -------------------------------------------------
    def preview(self):
        """Run SOR on the current working set and mark outliers red."""
        if len(self.pts) < 3:
            return
        self.config(cursor="watch")
        self.update()
        keep, *_ = compute_sor(self.fx, self.fy, self.fz,
                               k=self.k_var.get(), sigma=self.s_var.get())
        self.keep = keep
        self._split_indices()
        self.update_counts()
        self.render()
        self.config(cursor="")

    def apply_delete(self):
        """Commit: drop the red points, then reframe on what remains."""
        if not self.idx_drop:
            messagebox.showinfo("Nothing marked",
                                "Run 'Preview SOR' first to mark outliers.")
            return
        # snapshot for undo
        self.history.append((self.pts, self.fx, self.fy, self.fz))
        keep_idx = self.idx_keep
        self.pts = [self.pts[i] for i in keep_idx]
        self.fx = [self.fx[i] for i in keep_idx]
        self.fy = [self.fy[i] for i in keep_idx]
        self.fz = [self.fz[i] for i in keep_idx]
        self.keep = [True] * len(self.pts)
        self.applied += 1
        self._fit()
        self.zoom = 1.0
        self._split_indices()
        self.update_counts()
        self.render()

    def reframe(self):
        self._fit()
        self.zoom = 1.0
        self.render()

    def undo(self):
        if not self.history:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        self.pts, self.fx, self.fy, self.fz = self.history.pop()
        self.keep = [True] * len(self.pts)
        self.applied = max(0, self.applied - 1)
        self._fit()
        self.zoom = 1.0
        self._split_indices()
        self.update_counts()
        self.render()

    def save(self):
        if not self.pts:
            messagebox.showinfo("Save", "Open a points3D.txt first.")
            return
        kept = [self.pts[i] for i in self.idx_keep]   # excludes pending red
        out = filedialog.asksaveasfilename(
            title="Save cleaned points3D.txt",
            defaultextension=".txt",
            initialfile=os.path.basename(self.default_out),
            initialdir=os.path.dirname(self.default_out),
            filetypes=[("COLMAP points3D", "*.txt")])
        if not out:
            return
        save_points3d(out, kept)
        messagebox.showinfo(
            "Saved",
            f"Wrote {len(kept)} points to:\n{out}\n\n"
            f"({self.orig_total - len(kept)} removed in "
            f"{self.applied} delete pass(es))")


    # ---- menu bar (File + Toolkit) -----------------------------------
    def _build_menu(self):
        bar = tk.Menu(self, tearoff=0)

        filem = tk.Menu(bar, tearoff=0)
        filem.add_command(label="Open points3D.txt ...", command=self.open_file)
        filem.add_command(label="Save cleaned points3D.txt ...",
                          command=self.save)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        bar.add_cascade(label="File", menu=filem)

        # Toolkit = home for helper scripts; add future tools here.
        toolm = tk.Menu(bar, tearoff=0)
        toolm.add_command(label="points3D.txt   ->   PLY",
                          command=lambda: self.tool_convert("txt2ply"))
        toolm.add_command(label="PLY   ->   points3D.txt",
                          command=lambda: self.tool_convert("ply2txt"))
        toolm.add_separator()
        toolm.add_command(label="(more tools later ...)", state="disabled")
        bar.add_cascade(label="Toolkit", menu=toolm)

        self.config(menu=bar)

    # ---- open a cloud + first SOR preview ----------------------------
    def open_file(self):
        p = filedialog.askopenfilename(
            title="Open COLMAP points3D.txt",
            filetypes=[("COLMAP points3D", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.default_out) or os.path.dirname(__file__))
        if not p:
            return
        try:
            pts = load_points3d(p)
        except Exception as e:
            messagebox.showerror("Open failed", str(e))
            return
        if not pts:
            messagebox.showerror("Open failed", "No points found in file.")
            return
        self.pts = pts
        self.fx = [float(q[0]) for q in pts]
        self.fy = [float(q[1]) for q in pts]
        self.fz = [float(q[2]) for q in pts]
        self.orig_total = len(pts)
        self.applied = 0
        self.history = []
        self.default_out = os.path.splitext(p)[0] + "_clean.txt"
        self.title(f"COLMAP point cleaner  -  {os.path.basename(p)}")

        self.config(cursor="watch")
        self.update()
        keep, *_ = compute_sor(self.fx, self.fy, self.fz,
                               k=self.k_var.get(), sigma=self.s_var.get())
        self.keep = keep
        self.config(cursor="")
        self._fit()
        self.zoom = 1.0
        self._split_indices()
        self.update_counts()
        self.render()

    # ---- Toolkit : format converters ---------------------------------
    def tool_convert(self, which):
        msgs = []
        try:
            if which == "txt2ply":
                src = filedialog.askopenfilename(
                    title="Input points3D.txt",
                    filetypes=[("COLMAP points3D", "*.txt"), ("All files", "*.*")])
                if not src:
                    return
                dst = filedialog.asksaveasfilename(
                    title="Save PLY", defaultextension=".ply",
                    initialfile=os.path.splitext(os.path.basename(src))[0]
                    + "_forcleaning.ply",
                    filetypes=[("PLY point cloud", "*.ply")])
                if not dst:
                    return
                points3d_to_ply(src, dst, msgs.append)
            else:
                src = filedialog.askopenfilename(
                    title="Input PLY",
                    filetypes=[("PLY point cloud", "*.ply"), ("All files", "*.*")])
                if not src:
                    return
                dst = filedialog.asksaveasfilename(
                    title="Save points3D.txt", defaultextension=".txt",
                    initialfile="points3D.txt",
                    filetypes=[("COLMAP points3D", "*.txt")])
                if not dst:
                    return
                ply_to_points3d(src, dst, msgs.append)
            messagebox.showinfo("Toolkit", "\n".join(msgs) or "Done.")
        except Exception as e:
            messagebox.showerror("Toolkit failed", str(e))


# ==========================================================================
#  Headless CLI :  python point_converter.py --sor in.txt out.txt [opts]
# ==========================================================================
def run_cli(argv):
    a = argv[argv.index("--sor") + 1:]
    if len(a) < 2:
        print("usage: --sor <in.txt> <out.txt> [--neighbors N] [--sigma S]")
        return 2
    src, dst = a[0], a[1]
    k, sigma = 6, 1.0
    if "--neighbors" in argv:
        k = int(argv[argv.index("--neighbors") + 1])
    if "--sigma" in argv:
        sigma = float(argv[argv.index("--sigma") + 1])

    pts = load_points3d(src)
    fx = [float(p[0]) for p in pts]
    fy = [float(p[1]) for p in pts]
    fz = [float(p[2]) for p in pts]
    print(f"Loaded {len(pts)} points. SOR k={k} sigma={sigma} ...")

    def prog(i, n):
        print(f"  {100 * i // n}%", end="\r")

    keep, _, thr, mean, std = compute_sor(fx, fy, fz, k, sigma, progress=prog)
    kept = [pts[i] for i, kp in enumerate(keep) if kp]
    save_points3d(dst, kept)
    removed = len(pts) - len(kept)
    print(f"\nRemoved {removed} ({100 * removed / len(pts):.1f}%). "
          f"Kept {len(kept)} -> {dst}")
    return 0


if __name__ == "__main__":
    if "--sor" in sys.argv:
        sys.exit(run_cli(sys.argv))
    App().mainloop()
