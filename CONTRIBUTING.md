# Contributing to NASA Astronaut Recognition System

Thank you for your interest in contributing to the NASA Astronaut Recognition System! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of computer vision and machine learning
- Familiarity with OpenCV, face_recognition, and DeepFace libraries

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/nasa-astronaut-recognition.git
   cd nasa-astronaut-recognition
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements_reconocimiento.txt
   ```

## 📝 How to Contribute

### Types of Contributions
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test cases
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations

### Contribution Process

1. **Create an Issue**
   - Check if the issue already exists
   - Provide detailed description
   - Use appropriate labels

2. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Changes**
   - Follow coding standards (PEP 8)
   - Add tests for new features
   - Update documentation
   - Ensure all tests pass

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🎯 Coding Standards

### Python Style Guide
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Code Example
```python
def analyze_emotional_state(emotions: dict, confidence: float) -> tuple[str, str]:
    """
    Analyze emotional state based on detected emotions.
    
    Args:
        emotions: Dictionary of emotion probabilities
        confidence: Confidence level of emotion detection
        
    Returns:
        Tuple of (state, description)
    """
    if not emotions or len(emotions) == 0:
        return "unknown", "No emotional data"
    
    # Implementation here...
    return state, description
```

### File Organization
- Keep related functions together
- Use clear module names
- Separate configuration from logic
- Document complex algorithms

## 🧪 Testing

### Test Requirements
- All new features must include tests
- Maintain at least 80% code coverage
- Test edge cases and error conditions
- Include integration tests for API endpoints

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test_camera.py

# Run with coverage
pytest --cov=reconocimiento_antonio
```

### Test Structure
```python
def test_face_recognition():
    """Test facial recognition functionality."""
    # Setup
    known_faces = load_test_faces()
    
    # Test
    result = recognize_faces(test_image, known_faces)
    
    # Assert
    assert result[0] == "antonio"
    assert result[1] > 0.8
```

## 📚 Documentation

### Documentation Standards
- Update README.md for user-facing changes
- Add docstrings to all new functions
- Update API documentation
- Include usage examples
- Document configuration options

### Documentation Types
- **User Documentation**: README.md, installation guides
- **Developer Documentation**: Code comments, docstrings
- **API Documentation**: Endpoint descriptions, data structures
- **Troubleshooting**: Common issues and solutions

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Test with latest version
3. Verify it's not a configuration issue

### Bug Report Template
```markdown
**Bug Description**
Brief description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.8.5]
- Camera: [e.g., Logitech C920]

**Additional Context**
Any other relevant information.
```

## ✨ Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Brief description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives**
Other ways to solve this problem.

**Additional Context**
Any other relevant information.
```

## 🔄 Pull Request Process

### PR Requirements
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Clear commit messages

### PR Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

## 🏷️ Labels and Milestones

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority: high`: High priority
- `priority: low`: Low priority

### Milestones
- `v1.0.0`: Initial release
- `v1.1.0`: Feature release
- `v1.0.1`: Bug fix release

## 🎯 Development Roadmap

### Short Term (1-3 months)
- [ ] Multi-person recognition
- [ ] Performance optimizations
- [ ] Mobile app integration
- [ ] Cloud deployment

### Long Term (6+ months)
- [ ] AI model improvements
- [ ] Advanced health monitoring
- [ ] Integration with NASA systems
- [ ] Real-time alert system

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the issue, not the person
- Help others learn and grow

### Communication
- Use clear, concise language
- Provide context for questions
- Be patient with newcomers
- Share knowledge and resources

## 📞 Getting Help

### Resources
- 📖 [Documentation](README.md)
- 🐛 [Issues](https://github.com/yourusername/nasa-astronaut-recognition/issues)
- 💬 [Discussions](https://github.com/yourusername/nasa-astronaut-recognition/discussions)
- 📧 Email: your.email@nasa.gov

### Mentorship
- Look for issues labeled `good first issue`
- Ask questions in discussions
- Request code reviews
- Share your learning journey

## 🏆 Recognition

### Contributors
We recognize all contributors in our:
- README.md contributors section
- Release notes
- Annual contributor highlights
- Special recognition for significant contributions

### Contribution Types
- **Code**: Bug fixes, features, optimizations
- **Documentation**: Guides, tutorials, API docs
- **Testing**: Test cases, bug reports
- **Community**: Helping others, mentoring

---

**Thank you for contributing to NASA's mission! 🚀**

*Together, we're advancing space exploration through technology.*
