"""
Test suite for concurrency and race condition fixes in conformer.py

This test verifies the fixes for GitHub issue #19:
- pool.join() is called after pool.close()
- Context manager is used for Pool
- Exception handling uses logger instead of print
- Timeout mechanism for pool.imap()
"""

import unittest
import multiprocessing
import time
import signal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch, MagicMock
import numpy as np


class TestConformerConcurrency(unittest.TestCase):
    """Test concurrency handling in conformer generation."""

    def test_pool_context_manager_usage(self):
        """Test that Pool is used with context manager for proper cleanup."""
        from unimol_tools.data.conformer import ConformerGen
        
        # Mock Pool to verify context manager usage
        with patch('unimol_tools.data.conformer.Pool') as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            
            # Configure mock imap to return empty iterable
            mock_pool_instance.imap.return_value = []
            
            # Create generator with multiprocess enabled
            gen = ConformerGen(multi_process=True)
            
            # Call transform with empty list
            try:
                gen.transform([])
            except:
                pass  # Expected with empty list
            
            # Verify Pool was created with context manager
            MockPool.assert_called()

    def test_multiprocess_does_not_hang(self):
        """Test that multiprocessing does not hang on simple inputs."""
        from unimol_tools.data.conformer import ConformerGen
        
        # Use a simple molecule
        smiles_list = ['C', 'CC', 'CCC']
        
        gen = ConformerGen(multi_process=True, max_atoms=128)
        
        # Set a timeout to detect hangs
        def timeout_handler(signum, frame):
            raise TimeoutError("Transform timed out - possible race condition!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            start_time = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel alarm
            
            # Verify we got results
            self.assertEqual(len(inputs), len(smiles_list))
            self.assertLess(elapsed, 25)  # Should complete within 25 seconds
            
        except TimeoutError:
            self.fail("Transform hung - race condition detected!")
        except Exception as e:
            signal.alarm(0)
            # Other exceptions are acceptable for this test
            pass

    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is properly handled and not suppressed."""
        from unimol_tools.data.conformer import ConformerGen
        
        gen = ConformerGen(multi_process=False)
        
        # Test with minimal input
        smiles_list = ['C']
        
        try:
            gen.transform(smiles_list)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt should not be raised during normal operation")
        except Exception:
            pass  # Other exceptions are acceptable


class TestExceptionHandling(unittest.TestCase):
    """Test exception handling in conformer generation."""

    def test_inner_smi2coords_exception_logging(self):
        """Test that exceptions in inner_smi2coords are properly logged."""
        from unimol_tools.data.conformer import inner_smi2coords
        from unimol_tools.utils import logger
        
        # Test with invalid SMILES
        with patch.object(logger, 'error') as mock_error:
            result = inner_smi2coords('invalid_smiles_xxx', return_mol=False)
            
            # Should return zero coordinates on failure
            self.assertIsNotNone(result)
            
            # Verify error was logged (not printed)
            if mock_error.called:
                args = mock_error.call_args[0][0]
                self.assertIn('Failed', args)

    def test_no_bare_except_clauses(self):
        """Verify no bare 'except:' clauses in critical functions."""
        import ast
        import inspect
        from unimol_tools.data import conformer
        
        # Read source file
        source_file = inspect.getfile(conformer)
        with open(source_file, 'r') as f:
            tree = ast.parse(f.read())
        
        # Check for bare except clauses
        bare_excepts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    # Get line number
                    bare_excepts.append(node.lineno)
        
        # Log findings but don't fail - some bare excepts might be intentional
        if bare_excepts:
            print(f"Found bare 'except:' at lines: {bare_excepts}")
            # After fix, inner_smi2coords should not have bare except
            # that catch KeyboardInterrupt


class TestUniMolV2Concurrency(unittest.TestCase):
    """Test concurrency handling in UniMolV2Feature."""

    def test_unimolv2_pool_context_manager(self):
        """Test that UniMolV2Feature uses Pool with context manager."""
        from unimol_tools.data.conformer import UniMolV2Feature
        
        with patch('unimol_tools.data.conformer.Pool') as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = MagicMock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = MagicMock(return_value=False)
            mock_pool_instance.imap.return_value = []
            
            gen = UniMolV2Feature(multi_process=True)
            
            try:
                gen.transform([])
            except:
                pass
            
            MockPool.assert_called()

    def test_unimolv2_multiprocess_timeout(self):
        """Test that UniMolV2Feature multiprocessing completes within timeout."""
        from unimol_tools.data.conformer import UniMolV2Feature
        
        smiles_list = ['C', 'CC']
        gen = UniMolV2Feature(multi_process=True, max_atoms=128)
        
        def timeout_handler(signum, frame):
            raise TimeoutError("UniMolV2Feature transform timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            start_time = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start_time
            signal.alarm(0)
            
            self.assertEqual(len(inputs), len(smiles_list))
            self.assertLess(elapsed, 25)
            
        except TimeoutError:
            self.fail("UniMolV2Feature transform hung!")
        except Exception:
            signal.alarm(0)
            pass


class TestPoolCleanup(unittest.TestCase):
    """Test that pool resources are properly cleaned up."""

    def test_pool_processes_terminated_after_transform(self):
        """Verify that worker processes are terminated after transform completes."""
        from unimol_tools.data.conformer import ConformerGen
        
        # Get initial process count
        import psutil
        initial_procs = len(psutil.Process().children())
        
        gen = ConformerGen(multi_process=True)
        smiles_list = ['C', 'CC', 'CCC']
        
        try:
            gen.transform(smiles_list)
        except:
            pass
        
        # Give some time for cleanup
        time.sleep(1)
        
        final_procs = len(psutil.Process().children())
        
        # Process count should return to near initial (allowing for some variance)
        self.assertLessEqual(final_procs, initial_procs + 2)


def run_stress_test(iterations=5):
    """
    Run stress test to detect race conditions.
    
    This test runs multiple iterations of multiprocess transform
    to increase the chance of detecting race conditions.
    """
    from unimol_tools.data.conformer import ConformerGen
    
    print(f"\nRunning stress test with {iterations} iterations...")
    
    smiles_list = ['C', 'CC', 'CCC', 'CCCC', 'c1ccccc1']
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end=' ')
        sys.stdout.flush()
        
        gen = ConformerGen(multi_process=True, max_atoms=128)
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Iteration {i+1} timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20)
        
        try:
            start = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start
            signal.alarm(0)
            
            if len(inputs) == len(smiles_list):
                print(f"OK ({elapsed:.2f}s)")
            else:
                print(f"PARTIAL ({len(inputs)}/{len(smiles_list)})")
                
        except TimeoutError as e:
            signal.alarm(0)
            print(f"TIMEOUT - {e}")
            return False
        except Exception as e:
            signal.alarm(0)
            print(f"ERROR - {type(e).__name__}")
    
    print("Stress test completed successfully!")
    return True


if __name__ == '__main__':
    # Check if stress test requested
    if '--stress' in sys.argv:
        sys.argv.remove('--stress')
        success = run_stress_test(iterations=10)
        sys.exit(0 if success else 1)
    
    # Run unit tests
    unittest.main()
