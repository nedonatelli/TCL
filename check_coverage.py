import json

with open('coverage.json') as f:
    data = json.load(f)

print('=== TIER 2 MODULE COVERAGE (AFTER NEW TESTS) ===\n')

# Tier 2 modules
tier2_modules = {
    'pytcl/mathematical_functions/signal_processing/detection.py': 'Signal Detection',
    'pytcl/plotting/coordinates.py': 'Coordinate Plotting'
}

for module_path, description in tier2_modules.items():
    if module_path in data['files']:
        file_data = data['files'][module_path]
        executed = sum(1 for line_data in file_data['executed_lines'] if line_data >= 0)
        total_lines = len(file_data['executed_lines'])
        coverage = (executed / total_lines * 100) if total_lines > 0 else 0
        print(f'{description:25} {executed:3}/{total_lines:3} lines ({coverage:5.1f}%)')

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

print(f'\n=== OVERALL COVERAGE ===')
print(f'Total: {total_executed:,}/{total_lines:,} lines ({overall_coverage:.1f}%)')
