#!/usr/bin/env python3
"""
CFAR Detection Example
======================

Demonstrate Cell-Averaging CFAR detection on a simulated radar signal.
"""

import numpy as np

from pytcl.mathematical_functions.signal_processing import (
    cfar_ca,
    cfar_go,
    cfar_os,
    threshold_factor,
)


def main():
    np.random.seed(42)

    # Simulate radar range profile
    n_cells = 500

    # Background noise (exponential distribution for Swerling targets)
    signal = np.random.exponential(scale=1.0, size=n_cells)

    # Add targets at specific range cells
    target_cells = [100, 250, 400]
    target_snr = [15.0, 8.0, 5.0]  # Linear SNR

    for cell, snr in zip(target_cells, target_snr):
        signal[cell] = snr

    # CFAR parameters
    guard_cells = 3
    ref_cells = 10
    pfa = 1e-4

    print("CFAR Detection Results")
    print("=" * 50)
    print(f"True targets at cells: {target_cells}")
    print(f"Target SNRs: {target_snr}")
    print(f"Pfa: {pfa}")
    print()

    # Cell-Averaging CFAR
    ca_result = cfar_ca(signal, guard_cells, ref_cells, pfa)
    print(f"CA-CFAR detections: {ca_result.detection_indices.tolist()}")

    # Greatest-Of CFAR
    go_result = cfar_go(signal, guard_cells, ref_cells, pfa)
    print(f"GO-CFAR detections: {go_result.detection_indices.tolist()}")

    # Order-Statistic CFAR (robust to interferers)
    k = int(0.75 * 2 * ref_cells)  # 75th percentile
    os_result = cfar_os(signal, guard_cells, ref_cells, pfa, k=k)
    print(f"OS-CFAR detections: {os_result.detection_indices.tolist()}")

    # Threshold factor
    alpha = threshold_factor(pfa, 2 * ref_cells, method="ca")
    print(f"\nThreshold factor (CA-CFAR): {alpha:.2f}")

    # Detection performance
    for name, result in [("CA", ca_result), ("GO", go_result), ("OS", os_result)]:
        detected = set(result.detection_indices.tolist())
        hits = len(detected.intersection(target_cells))
        false_alarms = len(detected) - hits
        print(f"{name}-CFAR: {hits}/3 targets detected, " f"{false_alarms} false alarms")


if __name__ == "__main__":
    main()
