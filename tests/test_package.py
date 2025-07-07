#!/usr/bin/env python3
"""
Basic tests for repo_intel package
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Test imports
def test_imports():
    """Test that all modules can be imported"""
    import repo_intel
    from repo_intel import cli, diff, llm, markdown_bundle, glue_bundle

    assert repo_intel.__version__
    assert hasattr(cli, 'main')
    assert hasattr(diff, 'GitDiffAnalyzer')
    assert hasattr(llm, 'create_llm_provider')
    assert hasattr(markdown_bundle, 'MarkdownBundler')
    assert hasattr(glue_bundle, 'GlueDocumentationGenerator')


def test_version():
    """Test version is accessible"""
    import repo_intel
    assert isinstance(repo_intel.__version__, str)
    assert len(repo_intel.__version__) > 0


def test_settings_loading():
    """Test settings can be loaded"""
    from repo_intel.settings import OPENAI, ANTHROPIC, LLM

    # These should be AttrDict objects
    assert hasattr(OPENAI, 'API_KEY')
    assert hasattr(ANTHROPIC, 'API_KEY')
    assert hasattr(LLM, 'PROVIDER')


def test_llm_provider_factory():
    """Test LLM provider factory without API keys"""
    from repo_intel.llm import get_available_providers

    # Should be able to check availability
    providers = get_available_providers()
    assert isinstance(providers, dict)
    assert 'openai' in providers
    assert 'anthropic' in providers
    assert 'local' in providers


def test_markdown_bundler_basic():
    """Test markdown bundler basic functionality"""
    from repo_intel.markdown_bundle import MarkdownBundler

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "test.py").write_text("print('hello')")
        (temp_path / "test.js").write_text("console.log('hello');")

        # Create bundler
        output_file = temp_path / "bundle.md"
        bundler = MarkdownBundler(
            directory_path=str(temp_path),
            output_file=str(output_file)
        )

        # Should not raise errors
        bundler.create_bundle()
        assert output_file.exists()

        content = output_file.read_text()
        assert "test.py" in content
        assert "test.js" in content


def test_git_diff_analyzer_init():
    """Test GitDiffAnalyzer initialization"""
    from repo_intel.diff import GitDiffAnalyzer

    analyzer = GitDiffAnalyzer()
    assert analyzer.repo_path == Path(".")
    assert analyzer.max_file_size > 0
    assert analyzer.analyses == []


def test_git_diff_analyzer_risk_scoring():
    """Test risk scoring logic"""
    from repo_intel.diff import GitDiffAnalyzer

    analyzer = GitDiffAnalyzer()

    # Test basic risk scoring
    risk_score, priority = analyzer.calculate_risk_score(
        "src/core/main.py", 100, 50, "M"
    )
    assert 1 <= risk_score <= 10
    assert priority in ['low', 'medium', 'high', 'critical']

    # High-risk file should score higher
    risk_score_high, priority_high = analyzer.calculate_risk_score(
        "src/core/security.py", 500, 200, "M"
    )
    assert risk_score_high >= risk_score


@patch('boto3.Session')
def test_glue_documentation_generator_init(mock_session):
    """Test GlueDocumentationGenerator initialization"""
    from repo_intel.glue_bundle import GlueDocumentationGenerator

    # Mock the boto3 session
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    mock_session_instance.client.return_value = MagicMock()

    generator = GlueDocumentationGenerator()
    assert generator.output_file.name == "glue_documentation.md"


def test_cli_help():
    """Test CLI help doesn't crash"""
    from repo_intel.cli import create_parser

    parser = create_parser()
    # Should not raise
    help_text = parser.format_help()
    assert "repo-intel" in help_text
    assert "diff-analyze" in help_text
    assert "markdown-bundle" in help_text
    assert "glue-document" in help_text


if __name__ == "__main__":
    pytest.main([__file__])
