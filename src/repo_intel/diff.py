#!/usr/bin/env python3
"""
Git Branch Diff Analyzer for LLM Code Review

This script compares two git branches file by file, analyzes each change
with an LLM, and generates a comprehensive code review report.
"""

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from .llm import create_llm_provider
from .settings import LLM, GIT, OUTPUT

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Represents the analysis of a single file change"""
    file_path: str
    change_type: str  # 'modified', 'added', 'deleted', 'renamed'
    lines_added: int
    lines_removed: int
    diff_size: int
    file_size: int
    risk_score: int  # 1-10 scale
    priority: str  # 'low', 'medium', 'high', 'critical'
    llm_feedback: str
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ReviewSummary:
    """Overall review summary"""
    total_files: int
    files_analyzed: int
    files_skipped: int
    high_risk_files: int
    timestamp: str
    branch_comparison: str
    overall_assessment: str
    llm_provider: Optional[str] = None


class GitDiffAnalyzer:
    def __init__(self, repo_path: str = ".", max_file_size: int = None, llm_provider=None):
        self.repo_path = Path(repo_path)
        self.max_file_size = max_file_size or LLM.DEFAULT_MAX_FILE_SIZE
        self.analyses: List[FileAnalysis] = []
        self.llm_provider = llm_provider

    def get_git_diff(self, base_branch: str, compare_branch: str) -> str:
        """Get the diff between two branches"""
        try:
            result = subprocess.run(
                ['git', 'diff', f'{base_branch}...{compare_branch}', '--name-status'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git diff failed: {e}")
            raise

    def get_file_diff(self, file_path: str, base_branch: str, compare_branch: str) -> str:
        """Get diff for a specific file"""
        try:
            result = subprocess.run(
                ['git', 'diff', f'{base_branch}...{compare_branch}', '--', file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get diff for {file_path}: {e}")
            return ""

    def get_current_file_content(self, file_path: str, branch: str) -> str:
        """Get current content of a file from a specific branch"""
        try:
            result = subprocess.run(
                ['git', 'show', f'{branch}:{file_path}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            # File might not exist in the branch (new file)
            return ""

    def parse_diff_summary(self, diff_output: str) -> List[Tuple[str, str]]:
        """Parse git diff --name-status output"""
        files = []
        for line in diff_output.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                status = parts[0]
                file_path = parts[1]
                files.append((status, file_path))
        return files

    def calculate_risk_score(self, file_path: str, lines_added: int, lines_removed: int,
                             change_type: str) -> Tuple[int, str]:
        """Calculate risk score and priority for a file change"""
        risk_score = 0

        # Base risk from change size
        total_changes = lines_added + lines_removed
        if total_changes > 500:
            risk_score += 4
        elif total_changes > 200:
            risk_score += 3
        elif total_changes > 50:
            risk_score += 2
        else:
            risk_score += 1

        # File type risk
        high_risk_extensions = {'.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.rs'}
        config_files = {'dockerfile', 'docker-compose', 'requirements.txt', 'package.json', 'pom.xml', 'cargo.toml'}

        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name.lower()

        if file_ext in high_risk_extensions:
            risk_score += 2
        elif any(config in file_name for config in config_files):
            risk_score += 3
        elif 'test' in file_path.lower():
            risk_score += 1

        # Directory risk
        high_risk_dirs = {'src', 'lib', 'core', 'api', 'server', 'database', 'config'}
        if any(risk_dir in file_path.lower() for risk_dir in high_risk_dirs):
            risk_score += 2

        # Change type risk
        if change_type == 'D':  # Deleted
            risk_score += 3
        elif change_type == 'A':  # Added
            risk_score += 1
        elif change_type.startswith('R'):  # Renamed
            risk_score += 2

        # Convert to priority
        if risk_score >= 8:
            priority = 'critical'
        elif risk_score >= 6:
            priority = 'high'
        elif risk_score >= 4:
            priority = 'medium'
        else:
            priority = 'low'

        return min(risk_score, 10), priority

    def analyze_file_with_llm(self, file_path: str, diff_content: str,
                              file_content: str, change_type: str) -> str:
        """
        Analyze a file with an LLM using the configured provider
        """
        if self.llm_provider:
            try:
                return self.llm_provider.analyze_code_change(
                    file_path, diff_content, file_content, change_type
                )
            except Exception as e:
                logger.error(f"LLM analysis failed for {file_path}: {e}")
                return f"LLM analysis failed: {str(e)}"
        else:
            # No LLM provider configured
            return "No LLM analysis - provider not configured"

    def should_skip_file(self, file_path: str, diff_content: str, file_content: str) -> Tuple[bool, str]:
        """Determine if a file should be skipped due to size or other criteria"""
        combined_size = len(diff_content) + len(file_content)

        if combined_size > self.max_file_size:
            return True, f"Combined size ({combined_size} bytes) exceeds limit ({self.max_file_size} bytes)"

        # Skip binary files
        binary_extensions = {'.bin', '.exe', '.dll', '.so', '.dylib', '.img', '.iso', '.zip', '.tar', '.gz', '.png',
                             '.jpg', '.jpeg', '.gif', '.pdf'}
        if Path(file_path).suffix.lower() in binary_extensions:
            return True, "Binary file"

        # Skip very large files even if diff is small
        if len(file_content) > self.max_file_size * 0.8:
            return True, f"File too large ({len(file_content)} bytes)"

        return False, ""

    def analyze_branch_diff(self, base_branch: str, compare_branch: str) -> None:
        """Analyze all changes between two branches"""
        logger.info(f"Analyzing diff between {base_branch} and {compare_branch}")

        # Get list of changed files
        diff_summary = self.get_git_diff(base_branch, compare_branch)
        changed_files = self.parse_diff_summary(diff_summary)

        logger.info(f"Found {len(changed_files)} changed files")

        for status, file_path in changed_files:
            logger.info(f"Processing {file_path} ({status})")

            # Get file diff and content
            diff_content = self.get_file_diff(file_path, base_branch, compare_branch)
            file_content = self.get_current_file_content(file_path, compare_branch)

            # Calculate basic metrics
            lines_added = len([_line for _line in diff_content.split('\n') if _line.startswith('+')])
            lines_removed = len([_line for _line in diff_content.split('\n') if _line.startswith('-')])

            # Check if file should be skipped
            should_skip, skip_reason = self.should_skip_file(file_path, diff_content, file_content)

            # Calculate risk score
            risk_score, priority = self.calculate_risk_score(
                file_path, lines_added, lines_removed, status
            )

            if should_skip:
                analysis = FileAnalysis(
                    file_path=file_path,
                    change_type=status,
                    lines_added=lines_added,
                    lines_removed=lines_removed,
                    diff_size=len(diff_content),
                    file_size=len(file_content),
                    risk_score=risk_score,
                    priority=priority,
                    llm_feedback="",
                    skipped=True,
                    skip_reason=skip_reason
                )
                logger.warning(f"Skipping {file_path}: {skip_reason}")
            else:
                # Analyze with LLM
                llm_feedback = self.analyze_file_with_llm(
                    file_path, diff_content, file_content, status
                )

                analysis = FileAnalysis(
                    file_path=file_path,
                    change_type=status,
                    lines_added=lines_added,
                    lines_removed=lines_removed,
                    diff_size=len(diff_content),
                    file_size=len(file_content),
                    risk_score=risk_score,
                    priority=priority,
                    llm_feedback=llm_feedback,
                    skipped=False
                )

            self.analyses.append(analysis)

    def generate_report(self, base_branch: str, compare_branch: str, output_dir: str = None) -> None:
        """Generate comprehensive review report"""
        output_dir = output_dir or OUTPUT.DEFAULT_DIR
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create summary
        total_files = len(self.analyses)
        files_analyzed = len([a for a in self.analyses if not a.skipped])
        files_skipped = len([a for a in self.analyses if a.skipped])
        high_risk_files = len([a for a in self.analyses if a.priority in ['high', 'critical']])

        llm_provider_name = type(self.llm_provider).__name__ if self.llm_provider else "None"

        summary = ReviewSummary(
            total_files=total_files,
            files_analyzed=files_analyzed,
            files_skipped=files_skipped,
            high_risk_files=high_risk_files,
            timestamp=datetime.now().isoformat(),
            branch_comparison=f"{base_branch}...{compare_branch}",
            overall_assessment="Generated by automated analysis",
            llm_provider=llm_provider_name
        )

        # Save summary
        with open(output_path / "summary.json", "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save detailed analysis
        with open(output_path / "detailed_analysis.json", "w") as f:
            json.dump([asdict(a) for a in self.analyses], f, indent=2)

        # Generate markdown report
        self.generate_markdown_report(output_path, summary)

        # Generate individual file reports
        self.generate_file_reports(output_path)

        logger.info(f"Review report generated in {output_path}")

    def generate_markdown_report(self, output_path: Path, summary: ReviewSummary) -> None:
        """Generate a markdown summary report"""
        with open(output_path / "README.md", "w") as f:
            f.write("# Code Review Report\n\n")
            f.write(f"**Branch Comparison:** {summary.branch_comparison}\n")
            f.write(f"**Generated:** {summary.timestamp}\n")
            f.write(f"**LLM Provider:** {summary.llm_provider}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total files changed: {summary.total_files}\n")
            f.write(f"- Files analyzed: {summary.files_analyzed}\n")
            f.write(f"- Files skipped: {summary.files_skipped}\n")
            f.write(f"- High-risk files: {summary.high_risk_files}\n\n")

            # Sort by priority and risk score
            sorted_analyses = sorted(
                [a for a in self.analyses if not a.skipped],
                key=lambda x: (x.priority == 'critical', x.priority == 'high', x.risk_score),
                reverse=True
            )

            f.write("## Files by Priority\n\n")
            for priority in ['critical', 'high', 'medium', 'low']:
                priority_files = [a for a in sorted_analyses if a.priority == priority]
                if priority_files:
                    f.write(f"### {priority.title()} Priority ({len(priority_files)} files)\n\n")
                    for analysis in priority_files:
                        f.write(f"- [{analysis.file_path}](files/{analysis.file_path.replace('/', '_')}.md) ")
                        f.write(
                            f"(Risk: {analysis.risk_score}/10, +{analysis.lines_added}/-{analysis.lines_removed})\n")
                    f.write("\n")

            # Skipped files
            skipped_files = [a for a in self.analyses if a.skipped]
            if skipped_files:
                f.write("## Skipped Files\n\n")
                for analysis in skipped_files:
                    f.write(f"- {analysis.file_path}: {analysis.skip_reason}\n")

    def generate_file_reports(self, output_path: Path) -> None:
        """Generate individual file analysis reports"""
        files_dir = output_path / "files"
        files_dir.mkdir(exist_ok=True)

        for analysis in self.analyses:
            if analysis.skipped:
                continue

            # Create safe filename
            safe_filename = analysis.file_path.replace('/', '_').replace('\\', '_')
            file_report_path = files_dir / f"{safe_filename}.md"

            with open(file_report_path, "w") as f:
                f.write(f"# {analysis.file_path}\n\n")
                f.write(f"**Change Type:** {analysis.change_type}\n")
                f.write(f"**Risk Score:** {analysis.risk_score}/10\n")
                f.write(f"**Priority:** {analysis.priority}\n")
                f.write(f"**Lines Added:** {analysis.lines_added}\n")
                f.write(f"**Lines Removed:** {analysis.lines_removed}\n\n")

                f.write("## Analysis\n\n")
                f.write(analysis.llm_feedback)
                f.write("\n")


def main():
    """Standalone script entry point"""
    parser = argparse.ArgumentParser(description="Analyze git branch differences with LLM")
    parser.add_argument("base_branch", help="Base branch to compare from")
    parser.add_argument("compare_branch", help="Branch to compare to")
    parser.add_argument("--repo-path", default=str(GIT.DEFAULT_REPO_PATH), help="Path to git repository")
    parser.add_argument("--max-file-size", type=int, default=LLM.DEFAULT_MAX_FILE_SIZE,
                        help="Maximum combined file size for LLM analysis")
    parser.add_argument("--output-dir", default=OUTPUT.DEFAULT_DIR,
                        help="Output directory for reports")
    parser.add_argument("--llm-provider", choices=['openai', 'anthropic', 'local'],
                        help="LLM provider to use")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup LLM provider
    llm_provider = None
    if not args.no_llm:
        try:
            llm_provider = create_llm_provider(args.llm_provider)
            logger.info(f"Using LLM provider: {type(llm_provider).__name__}")
        except Exception as e:
            logger.warning(f"Failed to setup LLM provider: {e}. Continuing without LLM analysis.")

    analyzer = GitDiffAnalyzer(args.repo_path, args.max_file_size, llm_provider)
    analyzer.analyze_branch_diff(args.base_branch, args.compare_branch)
    analyzer.generate_report(args.base_branch, args.compare_branch, args.output_dir)


if __name__ == "__main__":
    main()
