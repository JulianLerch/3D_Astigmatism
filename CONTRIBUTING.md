# Contributing to 3D_Astigmatism

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Describe the bug clearly with steps to reproduce
- Include Python version, OS, and relevant error messages
- Attach example data if possible (anonymized)

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Explain why this would be useful to other users

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YourUsername/3D_Astigmatism.git
   cd 3D_Astigmatism
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Make your changes**
   - Follow existing code style
   - Add docstrings for new functions
   - Keep functions focused and testable

5. **Add tests**
   ```bash
   # Create test file in tests/
   # Run tests
   pytest tests/ -v
   ```

6. **Format and lint**
   ```bash
   black *.py tests/
   flake8 *.py tests/ --max-line-length=120
   ```

7. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

8. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Python**: PEP 8 with 120 character line length
- **Formatting**: Use `black` for automatic formatting
- **Type Hints**: Add type annotations where possible
- **Docstrings**: Use Google-style docstrings

Example:
```python
def calculate_msd(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Calculate mean squared displacement.

    Args:
        trajectory: Array of shape (n_frames, n_dims) with positions
        dt: Time step in seconds

    Returns:
        MSD values for each lag time
    """
    # Implementation
    pass
```

## Testing Guidelines

- **Coverage**: Aim for >80% test coverage
- **Test Types**:
  - Unit tests for individual functions
  - Integration tests for workflows
  - Use fixtures for reusable test data
- **Naming**: `test_function_name_behavior`

## Documentation

- Update README.md if adding new features
- Add docstrings to all public functions
- Update CHANGELOG.md with your changes

## Areas for Contribution

High-priority areas:
- 🧪 More comprehensive test coverage
- 📚 Example Jupyter notebooks
- 🌐 Internationalization (translations)
- 🚀 Performance optimizations
- 📊 Additional visualization options
- 🔬 New diffusion models/classifiers

## Questions?

Feel free to open an issue for discussion before starting work on major changes.

Thank you for helping improve 3D_Astigmatism! 🙏
