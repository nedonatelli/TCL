import json

with open('coverage.json') as f:
    data = json.load(f)

print('=' * 70)
print('COVERAGE MEASUREMENT AFTER NEW TESTS')
print('=' * 70)

# Tier 2 modules - the target for improvement
tier2_modules = {
    'pytcl/mathematical_functions/signal_processing/detection.py': 'Signal Detection',
    'pytcl/plotting/coordinates.py': 'Coordinate Plotting'
}

print('\n📊 TIER 2 TARGET MODULES (Low Coverage Priority)\n')
tier2_total_executed = 0
tier2_total_lines = 0

for module_path, description in sorted(tier2_modules.items()):
    if module_path in data['files']:
        file_data = data['files'][module_path]
        executed = sum(1 for line_data in file_data['executed_lines'] if line_data >= 0)
        total_lines = len(file_data['executed_lines'])
        coverage = (executed / total_lines * 100) if total_lines > 0 else 0
        
        tier2_total_executed += executed
        tier2_total_lines += total_lines
        
        status = "✅" if coverage >= 60 else "⚠️ " if coverage >= 45 else "❌"
        print(f"  {status} {description:25} {executed:3}/{total_lines:3} lines ({coverage:5.1f}%)")
    else:
        print(f"  ❌ {description:25} FILE NOT FOUND")

tier2_coverage = (tier2_total_executed / tier2_total_lines * 100) if tier2_total_lines > 0 else 0
print(f"\n  {'━' * 60}")
print(f"  🎯 Tier 2 Combined:          {tier2_total_executed:3}/{tier2_total_lines:3} lines ({tier2_coverage:5.1f}%)")

# Overall coverage
total_executed = sum(
    sum(1 for line_data in file['executed_lines'] if line_data >= 0)
    for file in data['files'].values()
)
total_lines = sum(
    len(file['executed_lines'])
    for file in data['files'].values()
)
overall_coverage = (total_executed / total_lines * 100) if total_lines > 0 else 0

print(f'\n📈 OVERALL PROJECT COVERAGE\n')
print(f'  Total covered:           {total_executed:,}/{total_lines:,} lines ({overall_coverage:.1f}%)')

print(f'\n' + '=' * 70)

# Show improvement comparison
print('\n📊 IMPROVEMENT SUMMARY\n')
print(f'  Signal Detection Module:  47.0% baseline → {(sum(1 for line_data in data["files"]["pytcl/mathematical_functions/signal_processing/detection.py"]["executed_lines"] if line_data >= 0)/len(data["files"]["pytcl/mathematical_functions/signal_processing/detection.py"]["executed_lines"]) * 100):.1f}% current')
print(f'  Coordinate Plotting:      40.4% baseline → {(sum(1 for line_data in data["files"]["pytcl/plotting/coordinates.py"]["executed_lines"] if line_data >= 0)/len(data["files"]["pytcl/plotting/coordinates.py"]["executed_lines"]) * 100):.1f}% current')
print(f'\n' + '=' * 70)
