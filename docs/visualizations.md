# Visualization guide

[Scene gallery](https://rtwotwo.github.io/DriveCoFi/#evidence) · [Attention explorer](https://rtwotwo.github.io/DriveCoFi/#attention) · [CSA explorer](https://rtwotwo.github.io/DriveCoFi/#augmentation)

The website provides standalone images and interactive trajectory plots. Selecting
a gallery image opens the image itself at a larger size.

## Planning gallery

The gallery contains **16 paired scenes**: three straight-driving, three left-turn,
three right-turn and seven additional maneuver examples from Appendix C. The scene
selector changes both planners together. Each planner generates and scores its own
candidates.

- Blue curves: generated candidate trajectories.
- Yellow path and waypoints: selected trajectory.
- White path, where present: ground-truth reference.
- Insets in the bird's-eye images enlarge the local trajectory geometry.

The selected examples illustrate lane following and turn geometry. Their scene
selection is qualitative; they are not a shared-candidate ranking benchmark.

## Camera-register attention

Choose front, left, right or rear camera, then RGB, an individual register, or the
mean map. There are **48 individual panels**: four cameras × two planners × six
views. The extracted panels retain the source image resolution. Warmer colors show
higher displayed attention. The displayed register indices are 0, 4, 8 and 12.

## Saved CSA trajectories

Five cached examples contain the original 64 proposals, 192 augmentation entries,
a saved selection and a reference trajectory. The explorer uses the stored x/y
coordinates rounded to four decimal places. It does not regenerate or rescore them.

Both axes use the same scale in meters, with the current ego pose at the origin.
Forward is upward and left is positive lateral position. Layer switches control
the displayed candidate sets. The time slider reveals one to eight future poses
at 0.5-second intervals; it does not animate a newly simulated vehicle or camera.
The selected path's visible waypoints are numbered.

CSA entries may include seed fallbacks. The caches do not provide a per-slot
validity/retry trace, so the explorer does not infer which entries were rejected or
label individual entries as collision-free.

## Generate new attention maps

With data and a checkpoint prepared, the release provides a per-scene attention
visualizer. Its dataset-root argument points to the directory containing `dataset/`:

```bash
python viz/viz_reg2patch_attention_single.py \
  --repo_root "$(pwd)" \
  --dataset_root /path/to/navsim_root \
  --checkpoint_path /path/to/checkpoint.ckpt \
  --agent_name more_ccs \
  --split navtest \
  --token YOUR_SCENE_TOKEN \
  --output_dir /path/to/visualizations \
  --dpi 300
```

Use `--csv /path/to/tokens.csv` for a list with a `token` column. Keep the register
count and checkpoint configuration consistent. The separate
`viz_reg2patch_attention_all.py` computes aggregate register cosine-similarity
statistics; it serves a different analysis from these scene heatmaps.

The trajectory visualization entry point is [viz_bev_traj.py](../viz/viz_bev_traj.py).
Run `python viz/viz_bev_traj.py --help` to inspect its camera, bird's-eye and token
selection options for your evaluation output.

## Asset provenance

| Website material | Source |
| --- | --- |
| Nine maneuver pairs | Current ICLR appendix, `Figures/cases/` |
| Seven additional pairs | Current ICLR appendix, `Figures/attentions/` trajectory panels |
| 48 attention panels | Embedded RGB/heatmap images from the appendix's four camera strips |
| Five CSA sets | Existing `viz/output/exp_arg_traj256/cache/*.npz` arrays |
| Results explorer | Main Tables 2–4 and appendix Tables B.4–B.5 |

The original per-scene figure sources are exported to WebP for direct browser
display; the site does not embed PDF pages. Original files remain in the research
workspace. [sources.json](../site/data/sources.json) records source filenames,
SHA-256 hashes and conversion operations. The gallery uses
[gallery.json](../site/data/gallery.json), and the trajectory explorer uses
[csa.json](../site/data/csa.json).

To preview the site from the repository root:

```bash
python -m http.server 8000 --directory site
```

Open `http://localhost:8000`. An HTTP server is required for loading JSON datasets.

To regenerate the web assets from the research figure directory and saved caches:

```bash
pip install pypdf pypdfium2 pillow numpy
python tools/export_paper_gallery.py \
  --figures /path/to/appendix/Figures \
  --csa-cache /path/to/exp_arg_traj256/cache
python tools/export_readme_previews.py
python tools/check_site.py
```

The exporters create WebP images, JSON trajectory datasets and SVG previews. The
original research figure directory and NPZ caches are inputs to this conversion;
the project site loads the exported browser assets directly.
