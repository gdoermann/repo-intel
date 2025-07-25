"""
LLM Integration for Repo Intel

This module provides LLM integration using environment-based configuration.
Enhanced with merge request summary generation capabilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import requests

from .settings import OPENAI, ANTHROPIC

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def analyze_code_change(self, file_path: str, diff_content: str,
                            file_content: str, change_type: str) -> str:
        """Analyze code change and return feedback"""
        pass

    def generate_mr_summary(self, context: str) -> str:
        """Generate a merge request summary. Default implementation uses analyze_code_change."""
        prompt = f"""
Based on the following git branch comparison data, write a concise 2-3 sentence summary 
suitable for a merge request description. Focus on:
- What functionality was changed, added, or removed
- The scope and impact of changes
- Any notable improvements or fixes

Write in a professional tone suitable for team review. Avoid technical jargon where possible.

Context:
{context}

Provide ONLY the summary paragraph, no additional formatting or explanations.
"""

        try:
            return self.analyze_code_change(
                "MERGE_REQUEST_SUMMARY", prompt, "", "SUMMARY"
            ).strip()
        except Exception as e:
            logger.error(f"Failed to generate MR summary: {e}")
            return "Unable to generate summary - manual description required."


class OpenAIProvider(LLMProvider):
    """OpenAI API integration using environment settings"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 organization: Optional[str] = None, temperature: Optional[float] = None):
        self.api_key = api_key or OPENAI.API_KEY
        self.model = model or OPENAI.MODEL
        self.organization = organization or OPENAI.ORGANIZATION
        self.temperature = temperature or OPENAI.TEMPERATURE
        self.base_url = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    def analyze_code_change(self, file_path: str, diff_content: str,
                            file_content: str, change_type: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        prompt = self._create_review_prompt(file_path, diff_content, file_content, change_type)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are an expert code reviewer. "
                            "Provide detailed, constructive feedback on code changes."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": self.temperature
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=OPENAI.TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            return f"Error analyzing with OpenAI: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI: {e}")
            return f"Error analyzing with OpenAI: {str(e)}"

    def generate_mr_summary(self, context: str) -> str:
        """Generate merge request summary using OpenAI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        prompt = self._create_mr_summary_prompt(context)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are a technical writer specializing in clear, concise summaries of code changes for development teams."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,  # Shorter for summaries
            "temperature": 0.3  # Lower temperature for consistency
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=OPENAI.TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Failed to generate OpenAI MR summary: {e}")
            return super().generate_mr_summary(context)

    def _create_review_prompt(self, file_path: str, diff_content: str,
                              file_content: str, change_type: str) -> str:
        return f"""
Please review this code change:

File: {file_path}
Change Type: {change_type}

Diff:
```diff
{diff_content}
```

Current File Content:
```
{file_content}
```

Please provide:
1. **Summary**: Brief description of what changed
2. **Issues**: Potential bugs, security concerns, or code quality issues
3. **Suggestions**: Specific improvements or best practices
4. **Risk Assessment**: Low/Medium/High risk level with reasoning
5. **Approval**: Approve/Needs Work/Reject with brief explanation

Focus on: correctness, security, performance, maintainability, and adherence to best practices.
"""

    def _create_mr_summary_prompt(self, context: str) -> str:
        return f"""
Based on the following git branch comparison, write a concise 2-3 sentence summary for a merge request description.

Requirements:
- Focus on what functionality was changed, added, or removed
- Mention the scope and impact of changes
- Use professional language suitable for team review
- Avoid excessive technical details
- Highlight any significant improvements or fixes

Context:
{context}

Write ONLY the summary paragraph, no formatting or additional text.
"""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API integration using environment settings"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key or ANTHROPIC.API_KEY
        self.model = model or ANTHROPIC.MODEL
        self.base_url = base_url or ANTHROPIC.BASE_URL

        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

    def analyze_code_change(self, file_path: str, diff_content: str,
                            file_content: str, change_type: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        prompt = self._create_review_prompt(file_path, diff_content, file_content, change_type)

        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()

            result = response.json()
            return result["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}")
            return f"Error analyzing with Anthropic: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error with Anthropic: {e}")
            return f"Error analyzing with Anthropic: {str(e)}"

    def generate_mr_summary(self, context: str) -> str:
        """Generate merge request summary using Anthropic Claude"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        prompt = self._create_mr_summary_prompt(context)

        payload = {
            "model": self.model,
            "max_tokens": 200,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()

            result = response.json()
            return result["content"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Failed to generate Anthropic MR summary: {e}")
            return super().generate_mr_summary(context)

    def _create_review_prompt(self, file_path: str, diff_content: str,
                              file_content: str, change_type: str) -> str:
        return f"""
I need you to perform a thorough code review of this change:

File: {file_path}
Change Type: {change_type}

Diff:
```diff
{diff_content}
```

Current File Content:
```
{file_content}
```

Please provide a structured review with:

1. **Change Summary**: What was modified and why
2. **Code Quality**: Issues with style, structure, or best practices
3. **Security Review**: Potential vulnerabilities or security concerns
4. **Performance Impact**: Any performance implications
5. **Testing Considerations**: What should be tested
6. **Risk Level**: Low/Medium/High with justification
7. **Recommendation**: Approve/Request Changes/Reject

Be specific and actionable in your feedback.
"""

    def _create_mr_summary_prompt(self, context: str) -> str:
        return f"""
Based on the git branch comparison data below, write a professional 2-3 sentence summary suitable for a merge request description.

Focus on:
- What functionality was changed, added, or removed
- The scope and impact of changes  
- Any notable improvements, bug fixes, or new features

Write in a clear, professional tone suitable for development team review.

Context:
{context}

Provide only the summary paragraph with no additional formatting.
"""


class LocalLLMProvider(LLMProvider):
    """Local LLM integration (e.g., Ollama, LM Studio)"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "codellama"):
        self.base_url = base_url
        self.model = model

    def analyze_code_change(self, file_path: str, diff_content: str,
                            file_content: str, change_type: str) -> str:
        # For Ollama API
        url = f"{self.base_url}/api/generate"

        prompt = self._create_review_prompt(file_path, diff_content, file_content, change_type)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            return result["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Local LLM API request failed: {e}")
            return f"Error analyzing with local LLM: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error with local LLM: {e}")
            return f"Error analyzing with local LLM: {str(e)}"

    def generate_mr_summary(self, context: str) -> str:
        """Generate merge request summary using local LLM"""
        url = f"{self.base_url}/api/generate"

        prompt = self._create_mr_summary_prompt(context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            return result["response"].strip()
        except Exception as e:
            logger.error(f"Failed to generate local LLM MR summary: {e}")
            return super().generate_mr_summary(context)

    def _create_review_prompt(self, file_path: str, diff_content: str,
                              file_content: str, change_type: str) -> str:
        return f"""
Code Review Request:

File: {file_path}
Change Type: {change_type}

Diff:
{diff_content}

Current File:
{file_content}

Please review this code change and provide:
1. Summary of changes
2. Potential issues or bugs
3. Security concerns
4. Performance implications
5. Suggestions for improvement
6. Overall assessment (approve/needs work/reject)

Be concise but thorough in your analysis.
"""

    def _create_mr_summary_prompt(self, context: str) -> str:
        return f"""
Based on the following git branch comparison data, write a brief 2-3 sentence summary for a merge request description.

Focus on what was changed and the impact. Be professional and concise.

Context:
{context}

Summary:
"""


def create_llm_provider(provider_type: Optional[str] = None, **kwargs) -> LLMProvider:
    """Factory function to create LLM provider instances using environment settings"""

    # If no provider specified, try to determine from available API keys
    if not provider_type:
        if OPENAI.API_KEY:
            provider_type = "openai"
        elif ANTHROPIC.API_KEY:
            provider_type = "anthropic"
        else:
            provider_type = "local"

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "local": LocalLLMProvider,
    }

    if provider_type not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")

    try:
        return providers[provider_type](**kwargs)
    except ValueError as e:
        logger.error(f"Failed to create {provider_type} provider: {e}")
        raise


def get_available_providers() -> Dict[str, bool]:
    """Check which LLM providers are available based on configuration"""
    return {
        "openai": bool(OPENAI.API_KEY),
        "anthropic": bool(ANTHROPIC.API_KEY),
        "local": True,  # Local is always available (assuming service is running)
    }


def auto_select_provider() -> str:
    """Automatically select the best available provider"""
    available = get_available_providers()

    # Priority order: OpenAI, Anthropic, Local
    for provider in ["openai", "anthropic", "local"]:
        if available[provider]:
            return provider

    raise RuntimeError("No LLM providers available")


# Legacy support for config file approach
def load_llm_config(config_path: str = "llm_config.json") -> Dict[str, Any]:
    """Load LLM configuration from JSON file (legacy support)"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return environment-based configuration
        provider = auto_select_provider()
        config = {"provider": provider}

        if provider == "openai":
            config.update({
                "model": OPENAI.MODEL,
                "api_key": OPENAI.API_KEY,
            })
        elif provider == "anthropic":
            config.update({
                "model": ANTHROPIC.MODEL,
                "api_key": ANTHROPIC.API_KEY,
            })

        return config


if __name__ == "__main__":
    # Example usage
    print("Available providers:", get_available_providers())

    try:
        provider = create_llm_provider()
        print(f"Created provider: {type(provider).__name__}")

        # Example analysis
        sample_diff = """
        +def calculate_total(items):
        +    total = 0
        +    for item in items:
        +        total += item.price
        +    return total
        """

        result = provider.analyze_code_change(
            "utils/calculator.py",
            sample_diff,
            "# File content here",
            "A"
        )

        print("Analysis result:", result)

        # Example MR summary
        sample_context = """
Branch Comparison: main -> feature/calculator

Commit Messages:
- Add price calculation utility
- Update test coverage

File Changes Summary:
- Core Logic Files: 2
- Test Files: 1

Code Statistics:
- Total Lines Added: 25
- Total Lines Removed: 3

Key File Changes:
utils/calculator.py (A, +15/-0)
tests/test_calculator.py (M, +10/-3)
"""

        summary = provider.generate_mr_summary(sample_context)
        print("MR Summary:", summary)

    except Exception as e:
        print(f"Error: {e}")
