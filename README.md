# Repo Intel 🔍

Repository Intelligence Tools - Analyze codebases with LLM assistance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repo Intel is a comprehensive toolkit for analyzing repositories, generating documentation, and performing AI-assisted code reviews. It provides three main tools:

- **Git Diff Analyzer** - Compare branches with LLM-powered code reviews
- **Markdown Bundler** - Combine code files into LLM-friendly markdown bundles
- **Glue Documenter** - Generate documentation for AWS Glue databases

## 🚀 Quick Start

### Installation

```bash
pip install repo-intel
```

### Basic Usage

```bash
# Analyze differences between branches
repo-intel diff-analyze main staging

# Create a markdown bundle of your codebase
repo-intel markdown-bundle src/ -o codebase.md

# Document AWS Glue databases
repo-intel glue-document -d my_database

# List available LLM providers
repo-intel list-providers
```

## 📋 Features

### Git Diff Analyzer
- **File-by-file analysis** that breaks down large diffs into manageable chunks
- **LLM integration** with OpenAI, Anthropic, and local models
- **Risk assessment** with automatic priority ranking
- **Smart filtering** to skip oversized or binary files
- **Comprehensive reporting** with both summary and detailed analysis

### Markdown Bundler
- **Flexible file inclusion** with configurable extensions
- **Smart exclusion patterns** for common directories (node_modules, .git, etc.)
- **Organized output** with table of contents and proper formatting
- **Markdown-only mode** for documentation bundling

### Glue Documenter
- **Complete AWS Glue documentation** for databases and tables
- **Schema documentation** with column details and types
- **Metadata extraction** including creation dates and parameters
- **Flexible filtering** to exclude specific databases or tables

## 🔧 Configuration

Repo Intel uses environment variables for configuration. Create a `.env` file or set these in your environment:

### LLM Configuration

```bash
# Choose your LLM provider
LLM_PROVIDER=openai  # 'openai', 'anthropic', 'local', or leave empty for auto-select

# OpenAI settings
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.3

# Anthropic settings  
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Local LLM settings (Ollama, etc.)
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=codellama
```

### AWS Configuration

```bash
AWS_REGION=us-west-2
AWS_PROFILE=default
```

### Other Settings

```bash
# Default output directory
OUTPUT_DEFAULT_DIR=repo_intel_output

# Maximum file size for LLM analysis (bytes)
LLM_DEFAULT_MAX_FILE_SIZE=250000

# Enable verbose logging
OUTPUT_VERBOSE=true
```

## 📖 Usage Examples

### Git Diff Analysis

```bash
# Basic branch comparison
repo-intel diff-analyze main feature/new-api

# Custom output directory and file size limit
repo-intel diff-analyze main staging \
    --output-dir my_review \
    --max-file-size 500000

# Skip LLM analysis for faster processing
repo-intel diff-analyze main staging --no-llm

# Force specific LLM provider
repo-intel diff-analyze main staging --llm-provider anthropic
```

### Markdown Bundling

```bash
# Bundle all code files
repo-intel markdown-bundle src/

# Bundle only markdown files
repo-intel markdown-bundle docs/ --markdown-only

# Exclude additional patterns
repo-intel markdown-bundle . --exclude __pycache__ .pytest_cache

# Custom output file
repo-intel markdown-bundle src/ -o my_codebase.md
```

### Glue Documentation

```bash
# Document all databases
repo-intel glue-document

# Document specific database
repo-intel glue-document -d my_database

# Use specific AWS profile and region
repo-intel glue-document --profile prod --region us-east-1

# Exclude specific databases or tables
repo-intel glue-document \
    --exclude-databases temp_db test_db \
    --exclude-tables temp_table
```

## 🏗️ Project Structure

```
repo-intel/
├── src/repo_intel/
│   ├── __init__.py          # Package version and metadata
│   ├── cli.py               # Main CLI interface
│   ├── diff.py              # Git diff analyzer
│   ├── llm.py               # LLM provider integrations
│   ├── markdown_bundle.py   # Markdown bundler
│   ├── glue_bundle.py       # AWS Glue documenter
│   └── settings.py          # Configuration management
├── tests/                   # Test suite
├── docs/                    # Documentation
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🔌 LLM Provider Setup

### OpenAI
1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set `OPENAI_API_KEY` environment variable
3. Choose your model (gpt-4, gpt-3.5-turbo, etc.)

### Anthropic
1. Get an API key from [Anthropic](https://console.anthropic.com/)
2. Set `ANTHROPIC_API_KEY` environment variable
3. Use Claude models for analysis

### Local LLM (Ollama)
1. Install [Ollama](https://ollama.ai/)
2. Pull a code model: `ollama pull codellama`
3. Start service: `ollama serve`
4. Configure `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL`

## 📊 Output Examples

### Git Diff Analysis Output
```
repo_intel_output/
├── README.md                # Summary with risk-ranked files
├── summary.json            # Machine-readable summary
├── detailed_analysis.json  # Complete analysis data
└── files/                  # Individual file reports
    ├── src_main.py.md
    ├── api_routes.py.md
    └── config_settings.py.md
```

### Risk Assessment
Files are automatically prioritized:
- **Critical** (8-10): Core changes requiring immediate attention
- **High** (6-7): Important changes needing careful review
- **Medium** (4-5): Standard changes requiring normal review
- **Low** (1-3): Minor changes needing quick review

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/gdoermann/repo-intel/issues)
- **Documentation**: [GitHub README](https://github.com/gdoermann/repo-intel#readme)
- **Discussions**: [GitHub Discussions](https://github.com/gdoermann/repo-intel/discussions)

## 🚧 Roadmap

- [ ] Support for more LLM providers (HuggingFace, local models)
- [ ] Integration with popular code review tools
- [ ] Advanced filtering and configuration options
- [ ] Web UI for report viewing
- [ ] CI/CD integration examples
- [ ] Plugin system for custom analyzers

---

**Made with ❤️ for developers who love clean, well-analyzed code**