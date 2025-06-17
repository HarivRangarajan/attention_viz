# ATTRIEVAL Testing Guide

This guide explains the comprehensive unit test suite for ATTRIEVAL (Attention-guided Retrieval for Long-Context Reasoning) and how to run and interpret the tests.

## 📋 Test Coverage

### Test Classes Overview

The test suite is organized into several comprehensive test classes:

#### 1. `TestAttrievelConfig`
Tests the configuration dataclass:
- **Default values**: Verifies all default configuration parameters
- **Custom values**: Tests initialization with custom parameters
- **Parameter validation**: Ensures valid ranges and types

#### 2. `TestAttrievelRetriever` 
Tests the main ATTRIEVAL implementation:
- **Initialization**: Constructor with default and custom configs
- **Context segmentation**: Fact extraction from text
- **Attention aggregation**: Multi-layer attention weight processing
- **Top-k token identification**: Attention peak detection
- **Attention sink filtering**: Noise removal
- **Fact scoring**: Mathematical correctness of scoring algorithm
- **Fact selection**: Top-N fact retrieval
- **Visualization**: Output formatting and display
- **Export functionality**: JSON result serialization
- **Integration testing**: End-to-end pipeline verification

#### 3. `TestEdgeCases`
Tests boundary conditions and error handling:
- **Empty inputs**: Behavior with empty context/responses
- **Minimal data**: Single tokens, short texts
- **Zero attention**: Handling of zero-weight scenarios
- **Boundary values**: Min/max parameter ranges

#### 4. `TestCreateAttrievelDemo`
Tests the demo function:
- **Model loading**: Mock model initialization
- **Pipeline execution**: Complete demo workflow
- **Result validation**: Output format verification

#### 5. `TestMathematicalCorrectness`
Verifies algorithm accuracy:
- **Attention normalization**: Preservation of probability distributions
- **Top-k selection**: Correctness of attention ranking
- **Fact scoring**: Mathematical formula implementation

## 🧪 Key Test Functions

### Context Segmentation Tests
```python
def test_segment_context_into_facts(self, retriever):
    """Test context segmentation into facts."""
```
- ✅ Validates fact structure (id, text, token_indices, etc.)
- ✅ Ensures minimum fact length requirements
- ✅ Verifies sequential token indexing
- ✅ Tests various punctuation patterns

### Attention Processing Tests  
```python
def test_aggregate_attention_weights(self, retriever):
    """Test attention weight aggregation across layers."""
```
- ✅ Validates output shape and dimensions
- ✅ Ensures non-negative, normalized values
- ✅ Verifies correct layer selection (last 25%)
- ✅ Tests mathematical aggregation accuracy

### Algorithm Correctness Tests
```python
def test_fact_scoring_mathematical_correctness(self):
    """Test that fact scoring follows the correct mathematical formula."""
```
- ✅ Manual calculation verification
- ✅ Formula implementation accuracy
- ✅ Numerical precision testing

## 🚀 Running the Tests

### Method 1: Using the Test Runner Script
```bash
cd attention_viz/
python run_attrieval_tests.py
```

### Method 2: Direct pytest Execution
```bash
cd attention_viz/
python -m pytest tests/test_attrieval.py -v
```

### Method 3: Specific Test Groups
```bash
# Run only configuration tests
pytest tests/test_attrieval.py::TestAttrievelConfig -v

# Run only mathematical correctness tests
pytest tests/test_attrieval.py::TestMathematicalCorrectness -v

# Run edge case tests
pytest tests/test_attrieval.py::TestEdgeCases -v
```

### Method 4: With Coverage Analysis
```bash
pip install pytest-cov
pytest tests/test_attrieval.py --cov=attention_viz.core.attrieval --cov-report=html
```

## 📊 Test Results Interpretation

### ✅ Successful Test Output
```
test_attrieval.py::TestAttrievelConfig::test_default_config PASSED    [ 5%]
test_attrieval.py::TestAttrievelConfig::test_custom_config PASSED     [10%]
test_attrieval.py::TestAttrievelRetriever::test_initialization PASSED [15%]
...
========================== 25 passed in 2.34s ==========================
```

### ❌ Failed Test Analysis
Failed tests indicate potential issues:
- **Assertion errors**: Logic or mathematical mistakes
- **Type errors**: Incorrect data structures
- **Mock failures**: Integration issues with dependencies

## 🔧 Test Configuration

### Mock Components
The tests use comprehensive mocking:
- **AttentionExtractor**: Mocked to generate predictable attention data
- **Model/Tokenizer**: Simulated transformer components
- **Torch operations**: Controlled tensor operations

### Test Data
Tests use controlled, predictable data:
- **Synthetic attention weights**: Normalized probability distributions
- **Medical context examples**: Realistic patient scenarios
- **Edge case scenarios**: Empty, minimal, and boundary inputs

## 🎯 Test Quality Metrics

### Coverage Goals
- **Function coverage**: 100% of public and private methods
- **Branch coverage**: All conditional paths tested
- **Edge case coverage**: Boundary conditions and error states
- **Integration coverage**: End-to-end pipeline testing

### Validation Types
1. **Structural validation**: Data structure integrity
2. **Mathematical validation**: Algorithm correctness
3. **Behavioral validation**: Expected output patterns
4. **Error handling validation**: Graceful failure modes

## 🔍 Debugging Failed Tests

### Common Issues and Solutions

#### 1. Mock Configuration Errors
```python
# Issue: Mock not returning expected data
# Solution: Check mock side_effect configuration
extractor.extract_attention_weights.side_effect = mock_extract_attention
```

#### 2. Numerical Precision Issues  
```python
# Issue: Floating point comparison failures
# Solution: Use numpy.testing.assert_allclose
np.testing.assert_allclose(result, expected, rtol=1e-10)
```

#### 3. Shape Mismatch Errors
```python
# Issue: Attention tensor dimensions don't match
# Solution: Verify mock attention generation logic
layer_weights = np.random.rand(num_heads, seq_len, seq_len)
```

### Debug Test Execution
```bash
# Run with maximum verbosity
pytest tests/test_attrieval.py -vvv --tb=long

# Run specific failing test
pytest tests/test_attrieval.py::TestAttrievelRetriever::test_score_facts -vvv

# Drop into debugger on failure
pytest tests/test_attrieval.py --pdb
```

## 📈 Extending the Test Suite

### Adding New Tests
When adding new ATTRIEVAL features:

1. **Create test class**: Group related functionality
2. **Add mock fixtures**: Provide controlled test data
3. **Test edge cases**: Include boundary and error conditions
4. **Verify mathematics**: Validate algorithmic correctness
5. **Integration test**: Test feature within full pipeline

### Test Template
```python
def test_new_feature(self, retriever):
    """Test description."""
    # Arrange: Set up test data
    input_data = create_test_data()
    
    # Act: Execute function under test
    result = retriever.new_feature(input_data)
    
    # Assert: Verify correctness
    assert result.property == expected_value
    assert len(result.items) == expected_count
```

## 🛡️ Test Maintenance

### Regular Test Updates
- **Algorithm changes**: Update tests when ATTRIEVAL implementation evolves
- **New edge cases**: Add tests for newly discovered boundary conditions
- **Performance tests**: Add benchmarks for critical functions
- **Integration tests**: Update when dependencies change

### Best Practices
1. **Descriptive test names**: Clear, specific function descriptions
2. **Isolated tests**: Each test independent of others
3. **Comprehensive mocking**: Control all external dependencies
4. **Clear assertions**: Specific, meaningful validation checks
5. **Documentation**: Explain complex test logic

---

This comprehensive test suite ensures the ATTRIEVAL implementation is reliable, mathematically correct, and handles edge cases gracefully. The tests serve as both verification and documentation of expected behavior. 