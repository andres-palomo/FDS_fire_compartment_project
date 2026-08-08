"""
Build HRRPUA Ramp from Experimental Mass-Loss Data
----------------------------------------------------
Reads the experimental mass-loss curve (mass vs time), differentiates
it to get mass loss rate (MLR), converts MLR to HRR using DELTA_H_CH
(the empirically corrected heat of combustion), then converts HRR to
HRRPUA using the MODEL pan area (not the physical pan area).

Outputs:
  - hrrpua_ramp.csv        : Time, HRR, HRRPUA, normalized F table
  - hrrpua_ramp_fds.txt     : ready-to-paste &RAMP lines for FDS
  - mass_loss_and_mlr.png  : experimental mass + derived MLR curve
  - hrrpua_ramp.png        : resulting HRRPUA ramp

Edit the variables in the "USER INPUTS" section before running.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# =====================================================================
# 1. USER INPUTS -- edit these before running
# =====================================================================
mass_loss_csv_path = "mass_loss_hep_160_150.csv"

# TODO: put your final mass-weighted DELTA_H_CH (Tewarson-derived) here.
# This MUST be paired with IDEAL = .FALSE. in &REAC, per your ΔH_ch /
# IDEAL discussion -- do NOT use ΔH_T (43,815 kJ/kg) here.
DELTA_H_CH_kJ_per_kg = 39540  # e.g. 41000.0  <-- fill in once derived

# Model pan area (NOT the physical pan area of 0.0256 m^2) -- this is
# what your FDS geometry (XB=2.000,2.150, 2.050,2.200) actually uses.
model_pan_area_m2 = 0.0225

# Must match the RAMP_Q ID referenced in your &SURF line exactly,
# e.g. RAMP_Q = 'hrrpua_hep_160_150' /
ramp_id = "hrrpua_hep_160_150"

# Light smoothing of the derivative (odd window length, polyorder < window)
# Set smooth_window = None to disable smoothing entirely.
smooth_window = 11
smooth_polyorder = 2

# =====================================================================
# 2. Load experimental mass-loss data
# =====================================================================
df = pd.read_csv(mass_loss_csv_path)
print("Columns found:", df.columns.tolist())
print(df.head())
print()

t_s = df["t_s"].values
mass_kg = df["mass_mean_kg"].values

print(f"Number of time points: {len(t_s)}")
print(f"Initial mass: {mass_kg[0]:.5f} kg")
print(f"Final mass:   {mass_kg[-1]:.5f} kg")
print(f"Total mass lost: {mass_kg[0] - mass_kg[-1]:.5f} kg")
print()

# =====================================================================
# 3. Differentiate mass to get mass loss rate (MLR), kg/s
# =====================================================================
# np.gradient gives d(mass)/dt; mass is decreasing so we negate it to
# get a positive burning rate.
mlr_raw_kg_s = -np.gradient(mass_kg, t_s)

if smooth_window is not None:
    mlr_kg_s = savgol_filter(mlr_raw_kg_s, window_length=smooth_window,
                              polyorder=smooth_polyorder)
    # smoothing can push tiny negative dips near zero -- clip those
    mlr_kg_s = np.clip(mlr_kg_s, 0, None)
else:
    mlr_kg_s = np.clip(mlr_raw_kg_s, 0, None)

plt.figure(figsize=(8, 5))
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(t_s, mass_kg, color="tab:brown", linewidth=1.5)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Fuel mass [kg]", color="tab:brown")
ax1.tick_params(axis="y", labelcolor="tab:brown")

ax2 = ax1.twinx()
ax2.plot(t_s, mlr_raw_kg_s, color="lightgray", linewidth=1, label="Raw MLR")
ax2.plot(t_s, mlr_kg_s, color="tab:orange", linewidth=1.5, label="Smoothed MLR")
ax2.set_ylabel("Mass loss rate [kg/s]", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.legend(loc="upper right")

plt.title("Experimental Mass Loss and Derived MLR")
plt.tight_layout()
plt.savefig("mass_loss_and_mlr.png", dpi=150)
print("Saved plot: mass_loss_and_mlr.png")
plt.show()

# =====================================================================
# 4. Convert MLR -> HRR -> HRRPUA
# =====================================================================
if DELTA_H_CH_kJ_per_kg is None:
    print()
    print("DELTA_H_CH_kJ_per_kg is not set -- fill it in before this step")
    print("produces meaningful HRR/HRRPUA values. Stopping here.")
else:
    # HRR [kW] = MLR [kg/s] * DELTA_H_CH [kJ/kg]  (kJ/s = kW)
    hrr_kW = mlr_kg_s * DELTA_H_CH_kJ_per_kg

    # HRRPUA [kW/m^2] = HRR [kW] / model pan area [m^2]
    hrrpua_kW_m2 = hrr_kW / model_pan_area_m2

    peak_hrrpua = hrrpua_kW_m2.max()
    print()
    print(f"Peak HRR:    {hrr_kW.max():.2f} kW")
    print(f"Peak HRRPUA: {peak_hrrpua:.2f} kW/m^2")

    # =================================================================
    # 5. Save the ramp table (Time, HRR, HRRPUA, normalized F)
    # =================================================================
    ramp_df = pd.DataFrame({
        "Time_s": t_s,
        "MLR_kg_s": mlr_kg_s,
        "HRR_kW": hrr_kW,
        "HRRPUA_kW_m2": hrrpua_kW_m2,
        "F_normalized": hrrpua_kW_m2 / peak_hrrpua,
    })
    ramp_df.to_csv("hrrpua_ramp.csv", index=False)
    print("Saved table: hrrpua_ramp.csv")

    # =================================================================
    # 6. Write ready-to-paste FDS &RAMP lines
    # =================================================================
    # FDS convention: SURF has HRRPUA = peak value, RAMP_Q references a
    # &RAMP table of (T, F) pairs where F is the FRACTION of peak (0-1).
    # To keep the FDS input file a manageable size, we don't write one
    # line per second -- we downsample to every Nth point. Adjust
    # downsample_step if you want a finer or coarser ramp.
    downsample_step = 2  # write every 2nd data point

    with open("hrrpua_ramp_fds.txt", "w") as f:
        f.write(f"! Peak HRRPUA = {peak_hrrpua:.3f} kW/m2 -- use this in your &SURF HRRPUA line\n")
        f.write(f"! This replaces your OLD &RAMP ID='{ramp_id}' lines -- delete those,\n")
        f.write(f"! keep your existing RAMP_Q = '{ramp_id}' reference in &SURF, and paste these in:\n\n")
        for i in range(0, len(t_s), downsample_step):
            f.write(f"&RAMP ID='{ramp_id}', T={t_s[i]:.3f}, F={ramp_df['F_normalized'].iloc[i]:.5f} /\n")
        # always include the final point even if the downsample step skips it
        if (len(t_s) - 1) % downsample_step != 0:
            f.write(f"&RAMP ID='{ramp_id}', T={t_s[-1]:.3f}, F={ramp_df['F_normalized'].iloc[-1]:.5f} /\n")

    print("Saved FDS ramp lines: hrrpua_ramp_fds.txt")

    # =================================================================
    # 7. Plot the resulting HRRPUA ramp
    # =================================================================
    plt.figure(figsize=(8, 5))
    plt.plot(t_s, hrrpua_kW_m2, color="tab:red", linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("HRRPUA [kW/m^2]")
    plt.title("Derived HRRPUA Ramp (from experimental MLR)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hrrpua_ramp.png", dpi=150)
    print("Saved plot: hrrpua_ramp.png")
    plt.show()

    # =================================================================
    # 8. Cross-check: total analytical energy this ramp implies
    # =================================================================
    from scipy.integrate import trapezoid
    total_energy_kJ = trapezoid(hrr_kW, t_s)
    print()
    print(f"Total analytical energy implied by this ramp: {total_energy_kJ:.1f} kJ")
    print("Use this as your analytical_energy_kJ target in hrr_energy_integration.py")