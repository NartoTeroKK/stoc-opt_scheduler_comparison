import os
import tempfile
import shutil
from src.data import process_and_save_all


def test_process_and_save_all_creates_expected_files():
    """Test that process_and_save_all creates the six expected .pt files."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a small config for fast testing
        config = {'seed': 42, 'augmentation': False}
        # Call the function with root set to the temporary directory
        process_and_save_all(config, root=tmpdir)

        processed_dir = os.path.join(tmpdir, 'processed')
        # List of expected files
        expected_files = [
            'breast_cancer_train.pt',
            'breast_cancer_val.pt',
            'breast_cancer_test.pt',
            'synthetic_train.pt',
            'synthetic_val.pt',
            'synthetic_test.pt'
        ]
        # Check that each file exists
        for fname in expected_files:
            fpath = os.path.join(processed_dir, fname)
            assert os.path.exists(fpath), f"Expected file {fpath} does not exist"
            # Optionally, check that the file is not empty
            assert os.path.getsize(fpath) > 0, f"File {fpath} is empty"