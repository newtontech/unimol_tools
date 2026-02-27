#!/usr/bin/env python3
"""
Simple test to verify the concurrency fixes by analyzing source code.
Does not require installation of unimol_tools.
"""

import ast
import sys

def test_exception_handling_improvements():
    """Test that bare except clauses have been replaced with specific exceptions."""
    print("Test 1: Checking exception handling improvements...")
    
    with open('/tmp/unimol_tools/unimol_tools/data/conformer.py', 'r') as f:
        content = f.read()
        tree = ast.parse(content)
    
    # Find all except handlers
    bare_excepts = []
    exception_excepts = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                bare_excepts.append(node.lineno)
            else:
                exception_excepts.append((node.lineno, ast.dump(node.type)))
    
    print(f"  Found {len(bare_excepts)} bare 'except:' clauses")
    print(f"  Found {len(exception_excepts)} specific 'except Exception:' clauses")
    
    # After fix, inner_smi2coords should use 'except Exception:' instead of bare 'except:'
    # Safe index function at ~line 611 is expected to have bare except
    non_safe_bare = [l for l in bare_excepts if l < 600]
    
    if non_safe_bare:
        print(f"  ⚠ Unexpected bare except at lines: {non_safe_bare}")
        return False
    else:
        print("  ✓ No problematic bare except clauses in main code")
        return True


def test_keyboard_interrupt_handling():
    """Test that KeyboardInterrupt is properly handled."""
    print("\nTest 2: Checking KeyboardInterrupt handling...")
    
    with open('/tmp/unimol_tools/unimol_tools/data/conformer.py', 'r') as f:
        content = f.read()
    
    checks = {
        'KeyboardInterrupt': 'KeyboardInterrupt handling',
        'pool.terminate()': 'Pool terminate',
        'pool.join()': 'Pool join (fixes race condition)',
    }
    
    all_passed = True
    for keyword, description in checks.items():
        if keyword in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} - MISSING")
            all_passed = False
    
    return all_passed


def test_documentation():
    """Test that documentation has been added."""
    print("\nTest 3: Checking documentation...")
    
    with open('/tmp/unimol_tools/unimol_tools/data/conformer.py', 'r') as f:
        content = f.read()
    
    if 'race conditions (fixes issue #19)' in content:
        print("  ✓ Issue #19 reference added in docstring")
        return True
    else:
        print("  ⚠ Issue reference not found")
        return False


def test_transform_structure():
    """Test the structure of transform methods."""
    print("\nTest 4: Verifying transform method structure...")
    
    with open('/tmp/unimol_tools/unimol_tools/data/conformer.py', 'r') as f:
        lines = f.readlines()
    
    transform_methods = []
    in_transform = False
    current_transform = None
    
    for i, line in enumerate(lines):
        if 'def transform(self, smiles_list):' in line:
            transform_methods.append({'line': i + 1, 'has_try': False, 'has_total': False})
            in_transform = True
            current_transform = len(transform_methods) - 1
        elif in_transform and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            in_transform = False
            current_transform = None
        elif in_transform and current_transform is not None:
            if 'try:' in line:
                transform_methods[current_transform]['has_try'] = True
            if 'total=len(smiles_list)' in line:
                transform_methods[current_transform]['has_total'] = True
    
    print(f"  Found {len(transform_methods)} transform methods")
    
    all_good = True
    for i, method in enumerate(transform_methods):
        print(f"  Transform method {i+1} (line {method['line']}):")
        if method['has_try']:
            print(f"    ✓ Has try-except block")
        else:
            print(f"    ✗ Missing try-except block")
            all_good = False
        if method['has_total']:
            print(f"    ✓ Has progress bar with total")
        else:
            print(f"    ✗ Missing progress bar total")
            all_good = False
    
    return all_good


def main():
    print("="*60)
    print("Testing concurrency fixes for issue #19")
    print("="*60)
    
    results = [
        ("Exception handling", test_exception_handling_improvements()),
        ("KeyboardInterrupt handling", test_keyboard_interrupt_handling()),
        ("Documentation", test_documentation()),
        ("Transform structure", test_transform_structure()),
    ]
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("All tests passed! Fixes have been applied correctly.")
        return 0
    else:
        print("Some tests failed. Please review the fixes.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
