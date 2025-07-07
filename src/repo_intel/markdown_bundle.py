#! /usr/bin/env python3
"""
Create a Markdown bundle of code files in a directory.
Supports configurable file extensions and exclusion patterns.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, Optional

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# File extension configurations with language mapping
EXTENSION_MAPPING: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".java": "java",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".txt": "text",
    ".sh": "bash",
    ".toml": "toml",
    ".tfvars": "terraform",
    ".yml": "yaml",
    ".yaml": "yaml",
}

# Directories and patterns to exclude
EXCLUDE_PATTERNS: Set[str] = {
    '.egg-info',
    '.aws-sam',
    '.git',
    '__pycache__',
    '__tests__',
    'node_modules',
    'venv',
    '.env',
    '.idea',
    '.yarn',
    ".yarn-lock",
    ".log",
    ".bkp",
    '.vscode',
    'package-lock.json',
    '.test.js',

}


class MarkdownBundler:
    """Class to handle the creation of markdown bundles from code files."""

    def __init__(
            self,
            directory_path: str,
            output_file: str = "bundle.md",
            extensions: Optional[Dict[str, str]] = None,
            exclude_patterns: Optional[Set[str]] = None,
            markdown: bool = False,
            no_header: bool = False
    ):
        self.directory = Path(directory_path)
        self.output_file = Path(output_file)
        self.extensions = extensions or EXTENSION_MAPPING
        self.exclude_patterns = exclude_patterns or EXCLUDE_PATTERNS
        self.markdown = markdown
        self.no_header = no_header

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded based on patterns or paths."""
        path_str = str(path)

        for pattern in self.exclude_patterns:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                # Handle absolute paths
                try:
                    # Check if the path is this pattern or a child of it
                    path.relative_to(pattern_path)
                    return True
                except ValueError:
                    continue
            else:
                # Handle relative paths and patterns
                if (pattern in path_str or
                        str(path.relative_to(self.directory)).startswith(str(pattern_path))):
                    return True
        return False

    def create_header(self) -> str:
        """Create the bundle header with metadata."""
        if self.markdown:
            return (
                f"# Markdown Bundle\n\n"
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Source directory: {self.directory.absolute()}\n\n"
                f"---\n\n"
            )
        else:
            return (
                f"# Code Bundle\n\n"
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Source directory: {self.directory.absolute()}\n\n"
                f"## Table of Contents\n\n"
            )

    def generate_toc(self, files: list[Path]) -> str:
        """Generate table of contents for the bundle."""
        toc = ""
        for file_path in files:
            relative_path = file_path.relative_to(self.directory)
            toc += f"- [{relative_path}](#{relative_path.as_posix().replace('/', '-')})\n"
        return toc

    def create_bundle(self) -> None:
        """Create a Markdown bundle of code files."""
        if not self.directory.is_dir():
            raise ValueError(f"Error: {self.directory} is not a valid directory.")

        # Collect valid files first
        valid_files = []
        for root, _, files in os.walk(self.directory):
            root_path = Path(root)
            if self.should_exclude(root_path):
                continue

            for file_name in files:
                file_path = root_path / file_name
                if (file_path.suffix in self.extensions and
                        not self.should_exclude(file_path)):
                    valid_files.append(file_path)

        # Sort files for consistent output
        valid_files.sort()

        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            if self.output_file.exists():
                logger.warning(f"Overwriting existing file: {self.output_file}")
                self.output_file.unlink()

            with self.output_file.open("w", encoding="utf-8") as markdown_file:
                # Write header and TOC
                if not self.no_header:
                    markdown_file.write(self.create_header())
                    markdown_file.write(self.generate_toc(valid_files))
                    markdown_file.write("\n---\n\n")

                # Write file contents
                if self.markdown:
                    valid_file = self.sort_files(valid_files)
                    if not valid_file:
                        logger.warning("No valid files found to include in the bundle.")
                        return

                for file_path in valid_files:
                    if not self.markdown:
                        relative_path = file_path.relative_to(self.directory)
                        markdown_file.write(f"\n## {relative_path}\n")
                        language = self.extensions[file_path.suffix]
                        markdown_file.write(f"```{language}\n")

                    try:
                        content = file_path.read_text(encoding="utf-8")
                        markdown_file.write(content)
                        if not content.endswith('\n'):
                            markdown_file.write('\n')
                        if self.markdown:
                            markdown_file.write("\n---\n\n")
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
                        markdown_file.write(f"# Error reading file: {e}\n")

                    if not self.markdown:
                        markdown_file.write("```\n")

                if self.markdown and self.no_header:
                    # Create it as a footer...
                    markdown_file.write(self.create_header())
            logger.info(f"Markdown bundle created: {self.output_file}")

        except Exception as e:
            logger.error(f"Error creating bundle: {e}")
            raise

    def sort_files(self, files: list[Path]) -> list[Path]:
        """
        Sort files for output ordering

        We should sort things as a tree structure, so that the output is more readable.
        Start with top level files, then add subdirectories and their files.

        Files should always come before subdirectories, and
        subdirectory files would be included before sub-subdirectories.
        :param files: input files
        :return: list of sorted files
        """
        if self.markdown:
            files = [file for file in files if file.suffix == '.md']
        # Sort files for consistent output
        # Group files by their parent directory
        files_by_dir = {}
        for file in files:
            relative_path = file.relative_to(self.directory)
            parent = relative_path.parent
            if parent not in files_by_dir:
                files_by_dir[parent] = []
            files_by_dir[parent] = files_by_dir[parent] + [file]

        # Sort files within each directory
        for dir_files in files_by_dir.values():
            dir_files.sort(key=lambda x: x.name)

        # Build final sorted list
        sorted_files = []
        # First add root files
        if Path('.') in files_by_dir:
            sorted_files.extend(files_by_dir[Path('.')])

        # Then add directory files in order of directory depth
        dirs = sorted(d for d in files_by_dir.keys() if d != Path('.'))
        for dir_path in sorted(dirs, key=lambda x: (len(x.parts), str(x))):
            sorted_files.extend(files_by_dir[dir_path])

        return sorted_files


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a Markdown bundle of code files in a directory."
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing code files."
    )
    parser.add_argument(
        "-o", "--output",
        default="bundle.md",
        help="Name of the output Markdown file."
    )
    parser.add_argument(
        "-m", "--markdown",
        action="store_true",
        help="Only include markdown and combine without code blocks."
    )
    parser.add_argument(
        '-e', '--exclude',
        nargs='*',
        default=[],
        help="Exclude directories or files based on patterns or paths"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not include the header in the bundle."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create a set of exclude patterns combining defaults with user-provided ones
    exclude_patterns = EXCLUDE_PATTERNS.copy()
    if args.exclude:
        exclude_patterns.update(args.exclude)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    bundler = MarkdownBundler(args.directory, args.output, markdown=args.markdown, no_header=args.no_header,
                              exclude_patterns=exclude_patterns)
    bundler.create_bundle()


if __name__ == "__main__":
    main()
