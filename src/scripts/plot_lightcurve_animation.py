"""
Generate an animation and final-frame snapshot of an example starspot system.

Output:
    figures/starspot_animation.mp4
    figures/starspot_animation_frame.pdf
"""
import numpy as np
import paths

from spotgp.lightcurve import LightcurveModel

# ── Example system parameters ───────────────────────────────────
model1 = LightcurveModel(
    peq=5.0,
    kappa=0.5,
    inc=np.radians(110),
    nspot=15,
    tem=3.0,
    tdec=3.0,
    alpha_max=0.1,
    lspot=15.0,
    tsim=40.0,
    tsamp=0.02,
)

# ── Produce animation + final frame ─────────────────────────────
outdir = paths.figures
anim1 = model1.animate_lightcurve(
    fps=15,
    duration=12.0,
    outfile=str(outdir / "starspot_animation_solar_dr.gif"),
    save_last_frame=str(outdir / "starspot_animation_solar_dr_frame.pdf"),
    dpi=150,
    show_spots=True,
    show_grid=True,
    show_params=True,
    show_dr=True,
    label_size=20
)

# ── Example system parameters ───────────────────────────────────
model2 = LightcurveModel(
    peq=5.0,
    kappa=-0.5,
    inc=np.radians(70),
    nspot=15,
    tem=3.0,
    tdec=3.0,
    alpha_max=0.1,
    lspot=15.0,
    tsim=40.0,
    tsamp=0.02,
)

# ── Produce animation + final frame ─────────────────────────────
anim2 = model2.animate_lightcurve(
    fps=15,
    duration=12.0,
    outfile=str(outdir / "starspot_animation_anti_dr.gif"),
    save_last_frame=str(outdir / "starspot_animation_anti_dr_frame.pdf"),
    dpi=150,
    show_spots=True,
    show_grid=True,
    show_params=True,
    show_dr=True,
    label_size=20
)
