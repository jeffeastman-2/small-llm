# Testing Guide

This document describes the comprehensive test suite for the Small-LLM project.

## Test Structure

The test suite is organized into several modules:

```
tests/
├── __init__.py
├── test_tokenizer.py      # Tokenization and vocabulary tests
├── test_dataset.py        # PyTorch dataset tests
├── test_model.py          # MiniGPT model tests
├── test_generation.py     # Text generation tests
├── test_data_loader.py    # PDF data loading tests (requires PyMuPDF)
└── test_integration.py    # End-to-end integration tests
```

## Running Tests

### Option 1: Using the built-in test runner

```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py --test test_tokenizer

# Quiet mode (minimal output)
python run_tests.py --quiet

# Verbose mode (detailed output)
python run_tests.py --verbose
```

### Option 2: Using pytest (recommended for development)

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests with coverage
python test_with_pytest.py

# Or run pytest directly
pytest tests/ -v --cov=src
```

### Option 3: Using standard unittest

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_tokenizer

# Run specific test method
python -m unittest tests.test_tokenizer.TestTokenizer.test_tokenize
```

## Test Categories

### 1. Unit Tests

**Tokenizer Tests (`test_tokenizer.py`)**
- Text tokenization functionality
- Vocabulary building and management
- Encoding/decoding consistency
- Vocabulary persistence (save/load)
- Edge cases (empty text, unknown words)

**Dataset Tests (`test_dataset.py`)**
- PyTorch dataset creation
- Data loading and batching
- Different context lengths
- Edge cases and bounds checking
- DataLoader compatibility

**Model Tests (`test_model.py`)**
- Model initialization and architecture
- Forward pass functionality
- Different model configurations
- Parameter counting
- Gradient flow
- Device compatibility
- Training/evaluation modes

**Generation Tests (`test_generation.py`)**
- Top-k and top-p sampling
- Text generation from seeds
- Temperature control
- Different generation parameters
- Edge cases (empty seeds, long seeds)
- Deterministic generation

### 2. Integration Tests

**Pipeline Tests (`test_integration.py`)**
- Complete training pipeline
- Model checkpointing and loading
- Vocabulary persistence
- End-to-end text generation
- Error handling
- Performance with different configurations

**Data Loading Tests (`test_data_loader.py`)**
- PDF text extraction
- Multi-file processing
- Text cleaning
- Error handling for missing files
- *Note: Requires PyMuPDF to run*

## Test Coverage

The test suite covers:

- ✅ **Core Functionality**: All major features tested
- ✅ **Edge Cases**: Empty inputs, boundary conditions
- ✅ **Error Handling**: Invalid inputs, missing files
- ✅ **Integration**: End-to-end workflows
- ✅ **Performance**: Large datasets and vocabularies
- ✅ **Device Compatibility**: CPU/GPU/MPS support
- ✅ **Reproducibility**: Deterministic behavior

## Test Requirements

### Core Requirements
- `torch>=1.13.0` - For model and tensor operations
- Standard library modules (unittest, tempfile, etc.)

### Optional Requirements
- `PyMuPDF>=1.20.0` - For PDF loading tests
- `pytest>=7.0.0` - For advanced test running
- `pytest-cov>=4.0.0` - For coverage reporting

### Installing Test Dependencies

```bash
# Install all dependencies including test tools
pip install -r requirements.txt

# Or install just the core dependencies
pip install torch
```

## Writing New Tests

When adding new features, follow these guidelines:

### 1. Test Organization
- Create tests in the appropriate module
- Use descriptive test method names
- Group related tests in test classes

### 2. Test Structure
```python
def test_feature_name(self):
    """Test description of what this test verifies."""
    # Arrange - set up test data
    input_data = "test input"
    
    # Act - call the function being tested
    result = function_to_test(input_data)
    
    # Assert - verify the results
    self.assertEqual(result, expected_output)
```

### 3. Best Practices
- Test both happy path and edge cases
- Use meaningful assertions
- Clean up resources in `tearDown()` if needed
- Use `subTest()` for parameterized tests
- Mock external dependencies when appropriate

### 4. Example Test

```python
def test_tokenize_empty_string(self):
    """Test tokenization of empty string."""
    result = tokenize("")
    self.assertEqual(result, [])
    
def test_tokenize_normal_text(self):
    """Test tokenization of normal text."""
    text = "Hello world!"
    result = tokenize(text)
    expected = ["hello", "world"]
    self.assertEqual(result, expected)
```

## Continuous Integration

The test suite is designed to work in CI environments:

- **Fast execution**: Core tests run in under 10 seconds
- **Minimal dependencies**: Can run with just PyTorch
- **Clear output**: Detailed failure reporting
- **Exit codes**: Proper success/failure signaling

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Missing PyMuPDF**
```bash
# PDF tests will be skipped if PyMuPDF is not installed
pip install PyMuPDF
```

**Memory Issues**
```bash
# Reduce batch sizes in integration tests if needed
# Tests are designed to use minimal memory
```

### Running Specific Test Subsets

```bash
# Run only fast tests (exclude slow integration tests)
python -m unittest discover tests -k "not integration"

# Run only unit tests
python run_tests.py --test test_tokenizer
python run_tests.py --test test_model
python run_tests.py --test test_dataset

# Run with different verbosity levels
python run_tests.py --quiet    # Minimal output
python run_tests.py --verbose  # Detailed output
```

## Performance Benchmarks

The test suite includes performance tests that verify:

- **Vocabulary building**: Handles 1000+ unique words
- **Dataset creation**: Processes 10,000+ token sequences
- **Model training**: Trains small models in seconds
- **Text generation**: Generates text in milliseconds

These benchmarks help ensure the codebase remains performant as features are added.

## Test Metrics

When running the full test suite, you should see:

- **50+ individual tests** across all modules
- **>95% code coverage** (with pytest-cov)
- **<30 seconds** total execution time
- **Zero failures** on supported platforms

This comprehensive test coverage ensures the reliability and maintainability of the Small-LLM codebase.
