#!/usr/bin/env python3
"""Analyze coverage JSON to identify gaps by tier."""

import json
import sys

with open('coverage.json') as f:
    data = json.load(f)

# Parse coverage data
files = data['files']
modules = {}

for filepath, info in files.items():
    if 'pytcl' not in filepath:
        continue
    
    # Extract module name
    parts = filepath.replace('.py', '').split('/')
    if 'pytcl' in parts:
        idx = parts.index('pytcl')
        module = '/'.join(parts[idx:])
    else:
        continue
    
    executed = len(info['executed_lines'])
    missing = len(info['missing_lines'])
    total = executed + missing
    
    if total > 0:
        coverage = executed / total * 100
    else:
        coverage = 100
    
    modules[module] = {
        'coverage': coverage,
        'executed': executed,
        'missing': missing,
        'total': total
    }

# Categorize by tier
tier_1 = []  # 0% coverage
tier_2 = []  # <50% coverage
tier_3 = []  # 50-70% coverage
tier_4 = []  # 70-90% coverage
tier_5 = []  # 90%+ coverage

for module, stats in modules.items():
    cov = stats['coverage']
    if cov == 0:
        tier_1.append((module, stats))
    elif cov < 50:
        tier_2.append((module, stats))
    elif cov < 70:
        tier_3.append((module, stats))
    elif cov < 90:
        tier_4.append((module, stats))
    else:
        tier_5.append((module, stats))

# Print results
print("=" * 80)
print("TIER 1: Zero Coverage (0%)")
print("=" * 80)
tier_1_statements = sum(m[1]['total'] for m in tier_1)
print(f"Modules: {len(tier_1)}, Total statements: {tier_1_statements}\n")
for module, stats in sorted(tier_1, key=lambda x: -x[1]['total'])[:15]:
    print(f"  {stats['total']:4d} statements | {module}")

print("\n" + "=" * 80)
print("TIER 2: Low Coverage (<50%)")
print("=" * 80)
tier_2_statements = sum(m[1]['total'] for m in tier_2)
print(f"Modules: {len(tier_2)}, Total statements: {tier_2_statements}\n")
for module, stats in sorted(tier_2, key=lambda x: -x[1]['total'])[:20]:
    print(f"{stats['coverage']:6.1f}% | {stats['total']:4d} statements | {module}")

print("\n" + "=" * 80)
print("TIER 3: Moderate Coverage (50-70%)")
print("=" * 80)
tier_3_statements = sum(m[1]['total'] for m in tier_3)
print(f"Modules: {len(tier_3)}, Total statements: {tier_3_statements}\n")
for module, stats in sorted(tier_3, key=lambda x: -x[1]['total'])[:20]:
    print(f"{stats['coverage']:6.1f}% | {stats['total']:4d} statements | {module}")

print("\n" + "=" * 80)
print("TIER 4: Good Coverage (70-90%)")
print("=" * 80)
tier_4_statements = sum(m[1]['total'] for m in tier_4)
print(f"Modules: {len(tier_4)}, Total statements: {tier_4_statements}\n")

print("\n" + "=" * 80)
print("TIER 5: Excellent Coverage (90%+)")
print("=" * 80)
tier_5_statements = sum(m[1]['total'] for m in tier_5)
print(f"Modules: {len(tier_5)}, Total statements: {tier_5_statements}\n")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Combine all tiers for summary stats
all_tiers = tier_1 + tier_2 + tier_3 + tier_4 + tier_5
total_statements = sum(m[1]['total'] for m in all_tiers)
total_executed = sum(m[1]['executed'] for m in all_tiers)
overall_coverage = total_executed / total_statements * 100 if total_statements > 0 else 100

print(f"Overall coverage: {overall_coverage:.1f}%")
print(f"Total statements: {total_statements}")
print(f"  Tier 1 (0%):       {tier_1_statements:5d} ({tier_1_statements/total_statements*100:5.1f}%)")
print(f"  Tier 2 (<50%):     {tier_2_statements:5d} ({tier_2_statements/total_statements*100:5.1f}%)")
print(f"  Tier 3 (50-70%):   {tier_3_statements:5d} ({tier_3_statements/total_statements*100:5.1f}%)")
print(f"  Tier 4 (70-90%):   {tier_4_statements:5d} ({tier_4_statements/total_statements*100:5.1f}%)")
print(f"  Tier 5 (90%+):     {tier_5_statements:5d} ({tier_5_statements/total_statements*100:5.1f}%)")
