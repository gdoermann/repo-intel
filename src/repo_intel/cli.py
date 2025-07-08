#!/usr/bin/env python3
"""
Repo Intel - Repository Analysis CLI

Main command-line interface for repository intelligence tools.
Enhanced with merge request summary generation.
"""

import argparse
import logging
import sys

from . import __version__
from .settings import OUTPUT, LLM, GIT, MARKDOWN, GLUE, AWS


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_diff_analyze(args):
    """Run git diff analysis"""
    from .diff import GitDiffAnalyzer
    from .llm import create_llm_provider, get_available_providers

    # Setup LLM provider
    llm_provider = None
    if not args.no_llm:
        try:
            available = get_available_providers()
            if not any(available.values()):
                logging.warning("No LLM providers configured. Running without LLM analysis.")
            else:
                llm_provider = create_llm_provider(args.llm_provider)
                logging.info(f"Using LLM provider: {type(llm_provider).__name__}")
        except Exception as e:
            logging.error(f"Failed to setup LLM provider: {e}")
            if args.require_llm:
                sys.exit(1)
            logging.warning("Continuing without LLM analysis")

    # Run analysis
    analyzer = GitDiffAnalyzer(
        repo_path=args.repo_path,
        max_file_size=args.max_file_size,
        llm_provider=llm_provider,
        force_regenerate=args.force
    )

    analyzer.analyze_branch_diff(args.base_branch, args.compare_branch, output_dir=args.output_dir)
    analyzer.generate_report(args.base_branch, args.compare_branch, args.output_dir)

    logging.info(f"Analysis complete. Report saved to: {args.output_dir}")


def cmd_mr_summary(args):
    """Generate merge request summary only"""
    from .diff import GitDiffAnalyzer
    from .llm import create_llm_provider, get_available_providers

    # Setup LLM provider
    llm_provider = None
    if not args.no_llm:
        try:
            available = get_available_providers()
            if not any(available.values()):
                logging.warning("No LLM providers configured. Generating basic summary.")
            else:
                llm_provider = create_llm_provider(args.llm_provider)
                logging.info(f"Using LLM provider: {type(llm_provider).__name__}")
        except Exception as e:
            logging.error(f"Failed to setup LLM provider: {e}")
            if args.require_llm:
                sys.exit(1)
            logging.warning("Generating basic summary without LLM")

    # Create analyzer
    analyzer = GitDiffAnalyzer(
        repo_path=args.repo_path,
        max_file_size=args.max_file_size,
        llm_provider=llm_provider,
        force_regenerate=False
    )

    # Run quick analysis (just get file statistics, no detailed LLM analysis)
    analyzer.analyze_branch_diff(args.base_branch, args.compare_branch, output_dir=None)

    # Generate summary
    summary = analyzer.generate_mr_summary(args.base_branch, args.compare_branch)

    # Output summary
    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
        logging.info(f"MR summary saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("MERGE REQUEST SUMMARY")
        print("=" * 60)
        print(f"\n{summary}\n")
        print("=" * 60)
        print("\nCopy the text above to your merge request description.")


def cmd_markdown_bundle(args):
    """Create markdown bundle"""
    from .markdown_bundle import MarkdownBundler

    bundler = MarkdownBundler(
        directory_path=args.directory,
        output_file=args.output,
        markdown=args.markdown_only,
        no_header=args.no_header,
        exclude_patterns=set(args.exclude) if args.exclude else None
    )

    bundler.create_bundle()
    logging.info(f"Markdown bundle created: {args.output}")


def cmd_glue_document(args):
    """Create Glue documentation"""
    from .glue_bundle import GlueDocumentationGenerator

    generator = GlueDocumentationGenerator(
        output_file=args.output,
        region=args.region,
        profile=args.profile,
        database_name=args.database,
        exclude_databases=set(args.exclude_databases) if args.exclude_databases else None,
        exclude_tables=set(args.exclude_tables) if args.exclude_tables else None,
        no_header=args.no_header
    )

    generator.create_documentation()
    logging.info(f"Glue documentation created: {args.output}")


def cmd_list_providers(args):
    """List available LLM providers"""
    from .llm import get_available_providers

    providers = get_available_providers()
    print("Available LLM providers:")
    for provider, available in providers.items():
        status = "✓ Available" if available else "✗ Not configured"
        print(f"  {provider}: {status}")


def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        prog='repo-intel',
        description='Repository Intelligence Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  repo-intel diff-analyze main staging
  repo-intel mr-summary main feature/new-api
  repo-intel markdown-bundle src/ -o code_bundle.md
  repo-intel glue-document -d my_database
  repo-intel list-providers
        """
    )

    parser.add_argument('--version', action='version', version=f'repo-intel {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=OUTPUT.VERBOSE, help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # Diff Analysis Command
    diff_parser = subparsers.add_parser('diff-analyze', aliases=['diff'], help='Analyze git branch differences')
    diff_parser.add_argument('base_branch', help='Base branch to compare from')
    diff_parser.add_argument('compare_branch', help='Branch to compare to')
    diff_parser.add_argument('--repo-path', default=str(GIT.DEFAULT_REPO_PATH),
                             help='Path to git repository')
    diff_parser.add_argument('--max-file-size', type=int, default=LLM.DEFAULT_MAX_FILE_SIZE,
                             help='Maximum combined file size for LLM analysis')
    diff_parser.add_argument('-o', '--output-dir', default=OUTPUT.DEFAULT_DIR,
                             help='Output directory for reports')
    diff_parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'local'],
                             help='Force specific LLM provider')
    diff_parser.add_argument('--no-llm', action='store_true',
                             help='Skip LLM analysis')
    diff_parser.add_argument('--require-llm', action='store_true',
                             help='Fail if no LLM provider is available')
    diff_parser.add_argument('-f', '--force', action='store_true',
                             help='Force re-analysis even if reports already exist')
    diff_parser.set_defaults(func=cmd_diff_analyze)

    # MR Summary Command
    mr_parser = subparsers.add_parser('mr-summary', aliases=['mrs'], help='Generate merge request summary')
    mr_parser.add_argument('base_branch', help='Base branch to compare from')
    mr_parser.add_argument('compare_branch', help='Branch to compare to')
    mr_parser.add_argument('--repo-path', default=str(GIT.DEFAULT_REPO_PATH),
                           help='Path to git repository')
    mr_parser.add_argument('--max-file-size', type=int, default=LLM.DEFAULT_MAX_FILE_SIZE,
                           help='Maximum combined file size for analysis')
    mr_parser.add_argument('-o', '--output', help='Output file for summary (prints to console if not specified)')
    mr_parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'local'],
                           help='Force specific LLM provider')
    mr_parser.add_argument('--no-llm', action='store_true',
                           help='Generate basic summary without LLM')
    mr_parser.add_argument('--require-llm', action='store_true',
                           help='Fail if no LLM provider is available')
    mr_parser.set_defaults(func=cmd_mr_summary)

    # Markdown Bundle Command
    bundle_parser = subparsers.add_parser('markdown-bundle', aliases=['mb'], help='Create markdown code bundle')
    bundle_parser.add_argument('directory', help='Directory to bundle')
    bundle_parser.add_argument('-o', '--output', default=MARKDOWN.DEFAULT_OUTPUT,
                               help='Output markdown file')
    bundle_parser.add_argument('-m', '--markdown-only', action='store_true',
                               help='Only include markdown files')
    bundle_parser.add_argument('--no-header', action='store_true',
                               help='Skip header in output')
    bundle_parser.add_argument('-e', '--exclude', nargs='*',
                               help='Additional patterns to exclude')
    bundle_parser.set_defaults(func=cmd_markdown_bundle)

    # Glue Documentation Command
    glue_parser = subparsers.add_parser('glue-document', aliases=['gb'], help='Document AWS Glue databases')
    glue_parser.add_argument('-o', '--output', default=GLUE.DEFAULT_OUTPUT,
                             help='Output markdown file')
    glue_parser.add_argument('-r', '--region', default=AWS.REGION,
                             help='AWS region')
    glue_parser.add_argument('-p', '--profile', default=AWS.PROFILE,
                             help='AWS profile')
    glue_parser.add_argument('-d', '--database', help='Specific database to document')
    glue_parser.add_argument('--exclude-databases', nargs='*',
                             default=GLUE.EXCLUDE_DATABASES,
                             help='Databases to exclude')
    glue_parser.add_argument('--exclude-tables', nargs='*',
                             default=GLUE.EXCLUDE_TABLES,
                             help='Tables to exclude')
    glue_parser.add_argument('--no-header', action='store_true',
                             help='Skip header in output')
    glue_parser.set_defaults(func=cmd_glue_document)

    # List Providers Command
    list_parser = subparsers.add_parser('list-providers', help='List available LLM providers')
    list_parser.set_defaults(func=cmd_list_providers)

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
