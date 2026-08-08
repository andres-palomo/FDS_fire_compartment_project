"""
HRR Integration & Energy Comparison
------------------------------------
Reads the FDS `_hrr.csv` output file, integrates the simulated heat
release rate (HRR) over time to get the total simulated energy released,
and compares it against the analytical / prescribed total energy.

Edit the variables in the "USER INPUTS" section below, then run this
script with:
    python hrr_energy_integration.py
"""

import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# =====================================================================
# 1. USER INPUTS -- edit these before running
# =====================================================================
hrr_csv_path = "hep_160_150_1_1_hrr.csv"
mass_csv_path = "hep_160_150_1_1_mass.csv"
analytical_energy_kJ = 39540*0.15  # e.g. 12345.6  <-- put your prescribed target here

# =====================================================================
# 2. Load the HRR CSV file
# =====================================================================
# FDS writes _hrr.csv with two header rows:
#   Row 1: units (s, kW, kW, ...)
#   Row 2: column names (Time, HRR, Q_RADI, ...)
# so we skip the units row and use the next row as column names.
hrr_df = pd.read_csv(hrr_csv_path, skiprows=1)

print("Columns found:", hrr_df.columns.tolist())
print(hrr_df.head())
print()

time_s = hrr_df["Time"].values
hrr_kW = hrr_df["HRR"].values

print(f"Number of time points: {len(time_s)}")
print(f"Simulation duration: {time_s[-1]:.1f} s")
print(f"Peak HRR: {hrr_kW.max():.1f} kW")
print()

# =====================================================================
# 3. Plot the simulated HRR curve
# =====================================================================
plt.figure(figsize=(8, 5))
plt.plot(time_s, hrr_kW, color="tab:red", linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel("HRR [kW]")
plt.title("Simulated Heat Release Rate (FDS)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hrr_curve.png", dpi=150)
print("Saved plot: hrr_curve.png")
plt.show()

# =====================================================================
# 4. Integrate HRR over time
# =====================================================================
# HRR is in kW = kJ/s and time is in s, so integrating HRR with respect
# to time gives the total energy released in kJ. We use the trapezoidal
# rule, the standard choice for this kind of time series.
simulated_energy_kJ = trapezoid(hrr_kW, time_s)
simulated_energy_MJ = simulated_energy_kJ / 1000

print()
print(f"Total simulated energy: {simulated_energy_kJ:.1f} kJ  ({simulated_energy_MJ:.3f} MJ)")
# =====================================================================
# 5. Cumulative energy released over time
# =====================================================================
# Useful to see WHEN a mismatch builds up, not just the final total.
cumulative_energy_kJ = np.zeros_like(time_s)
for i in range(1, len(time_s)):
    cumulative_energy_kJ[i] = trapezoid(hrr_kW[:i + 1], time_s[:i + 1])

plt.figure(figsize=(8, 5))
plt.plot(time_s, cumulative_energy_kJ, color="tab:blue", linewidth=1.5,
         label="Simulated (cumulative)")

if analytical_energy_kJ is not None:
    plt.axhline(analytical_energy_kJ, color="black", linestyle="--", linewidth=1.2,
                label=f"Analytical target ({analytical_energy_kJ:.1f} kJ)")

plt.xlabel("Time [s]")
plt.ylabel("Cumulative energy [kJ]")
plt.title("Cumulative Energy Released")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cumulative_energy.png", dpi=150)
print("Saved plot: cumulative_energy.png")
plt.show()

# =====================================================================
# 6. Compare simulated vs analytical total energy
# =====================================================================
# NOTE: if this is a partial test run (e.g. 30 s), analytical_energy_kJ
# must be the time-integrated prescribed ramp up to that same time, not
# the full-run target.
print()
if analytical_energy_kJ is None:
    print("Set analytical_energy_kJ at the top of this script to run the comparison.")
else:
    absolute_error_kJ = simulated_energy_kJ - analytical_energy_kJ
    percent_error = (absolute_error_kJ / analytical_energy_kJ) * 100

    print(f"Simulated energy:   {simulated_energy_kJ:10.1f} kJ")
    print(f"Analytical energy:  {analytical_energy_kJ:10.1f} kJ")
    print(f"Absolute error:     {absolute_error_kJ:10.1f} kJ")
    print(f"Percent error:      {percent_error:10.3f} %")

# =====================================================================
# 7. Cross-check: fuel mass burned and implied heat of combustion
# =====================================================================
# Integrating MLR_HEPTAN_SURROGATE (fuel mass loss rate, kg/s) over time
# gives total fuel mass burned. Dividing simulated energy by this mass
# gives the heat of combustion FDS effectively used -- compare this to
# your &REAC ΔH and your derived ΔH_ch from the handbook.
mlr_fuel_kg_s = hrr_df["MLR_HEPTAN_SURROGATE"].values

total_fuel_burned_kg = trapezoid(mlr_fuel_kg_s, time_s)
implied_dHc_kJ_per_kg = simulated_energy_kJ / total_fuel_burned_kg

print()
print(f"Total fuel burned (integrated MLR): {total_fuel_burned_kg:.5f} kg")
print(f"Implied heat of combustion:         {implied_dHc_kJ_per_kg:.1f} kJ/kg")
print("Compare this to the ΔH in &REAC and your mass-weighted ΔH_ch from Table A.40.")

# =====================================================================
# 8. Optional: sanity-check against the _mass.csv file
# =====================================================================
# _mass.csv tracks species mass currently IN THE DOMAIN (not cumulative
# mass burned), so it won't give total fuel consumed directly. It's
# useful just to confirm fuel species mass behaves sensibly over time.
mass_df = pd.read_csv(mass_csv_path, skiprows=1)

plt.figure(figsize=(8, 5))
plt.plot(mass_df["Time"], mass_df["HEPTAN_SURROGATE"], color="tab:green", linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel("Fuel species mass in domain [kg]")
plt.title("HEPTAN_SURROGATE Mass in Domain (gas phase)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fuel_mass_in_domain.png", dpi=150)
print("Saved plot: fuel_mass_in_domain.png")
plt.show()
