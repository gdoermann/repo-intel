#! /usr/bin/env python3
"""
Create a Markdown documentation of AWS Glue databases and tables.
This tool exports AWS Glue metadata to a single markdown file for LLM context.
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlueDocumentationGenerator:
    """Class to handle the creation of markdown documentation from AWS Glue resources."""

    def __init__(
            self,
            output_file: str = "glue_documentation.md",
            region: Optional[str] = None,
            profile: Optional[str] = None,
            database_name: Optional[str] = None,
            exclude_databases: Optional[Set[str]] = None,
            exclude_tables: Optional[Set[str]] = None,
            no_header: bool = False
    ):
        self.output_file = Path(output_file)
        self.region = region
        self.profile = profile
        self.database_name = database_name
        self.exclude_databases = exclude_databases or set()
        self.exclude_tables = exclude_tables or set()
        self.no_header = no_header

        # Initialize AWS session
        session_args = {}
        if profile:
            session_args['profile_name'] = profile
        if region:
            session_args['region_name'] = region

        self.session = boto3.Session(**session_args)
        self.glue_client = self.session.client('glue')

    def create_header(self) -> str:
        """Create the documentation header with metadata."""
        region = self.region or self.session.region_name or "default"
        return (
            f"# AWS Glue Documentation\n\n"
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"AWS Region: {region}\n"
            f"Profile: {self.profile or 'default'}\n\n"
            f"## Table of Contents\n\n"
        )

    def get_databases(self) -> List[Dict]:
        """Get a list of all Glue databases or a specific one."""
        if self.database_name:
            try:
                response = self.glue_client.get_database(Name=self.database_name)
                return [response['Database']]
            except Exception as e:
                logger.error(f"Error retrieving database {self.database_name}: {e}")
                return []
        else:
            databases = []
            paginator = self.glue_client.get_paginator('get_databases')
            for page in paginator.paginate():
                databases.extend(page['DatabaseList'])
            return databases

    def get_tables(self, database_name: str) -> List[Dict]:
        """Get a list of all tables in a database."""
        tables = []
        paginator = self.glue_client.get_paginator('get_tables')
        try:
            for page in paginator.paginate(DatabaseName=database_name):
                tables.extend(page['TableList'])
            return tables
        except Exception as e:
            logger.error(f"Error retrieving tables for database {database_name}: {e}")
            return []

    def format_column_type(self, column_type: str) -> str:
        """Format column type for better readability."""
        return f"`{column_type}`"

    def generate_table_schema(self, table: Dict) -> str:
        """Generate markdown for a table schema."""
        md = f"### {table['Name']}\n\n"

        # Add table description if available
        if 'Description' in table and table['Description']:
            md += f"{table['Description']}\n\n"

        # Add table metadata
        md += "**Metadata:**\n\n"
        md += f"- **Created:** {table.get('CreateTime', 'Unknown')}\n"
        md += f"- **Updated:** {table.get('UpdateTime', 'Unknown')}\n"
        if 'TableType' in table and table['TableType']:
            md += f"- **Type:** {table['TableType']}\n"
        if 'Location' in table['StorageDescriptor']:
            md += f"- **Location:** `{table['StorageDescriptor']['Location']}`\n"
        if 'Parameters' in table and table['Parameters']:
            md += "\n**Parameters:**\n\n"
            for key, value in table['Parameters'].items():
                md += f"- **{key}:** {value}\n"

        md += "\n**Schema:**\n\n"

        # Add column headers
        md += "| Column | Type | Description |\n"
        md += "|--------|------|-------------|\n"

        # Add column details
        for column in table['StorageDescriptor']['Columns']:
            description = column.get('Comment', '')
            md += f"| {column['Name']} | {self.format_column_type(column['Type'])} | {description} |\n"

        # Add partition columns if available
        if 'PartitionKeys' in table and table['PartitionKeys']:
            md += "\n**Partition Keys:**\n\n"
            md += "| Column | Type | Description |\n"
            md += "|--------|------|-------------|\n"

            for column in table['PartitionKeys']:
                description = column.get('Comment', '')
                md += f"| {column['Name']} | {self.format_column_type(column['Type'])} | {description} |\n"

        return md

    def generate_toc(self, databases: List[Dict], tables_by_db: Dict[str, List[Dict]]) -> str:
        """Generate table of contents for the documentation."""
        toc = ""

        for db in databases:
            db_name = db['Name']
            if db_name in self.exclude_databases:
                continue

            toc += f"- [{db_name}](#{db_name.lower().replace(' ', '-')})\n"

            if db_name in tables_by_db:
                for table in tables_by_db[db_name]:
                    table_name = table['Name']
                    if table_name in self.exclude_tables:
                        continue
                    toc += (f"  - [{table_name}](#{db_name.lower().replace(' ', '-')}"
                            f"-{table_name.lower().replace(' ', '-')})\n")

        return toc

    def create_documentation(self) -> None:
        """Create a Markdown documentation of Glue databases and tables."""
        try:
            # Get all databases or the specified one
            databases = self.get_databases()

            # Filter excluded databases
            databases = [db for db in databases if db['Name'] not in self.exclude_databases]

            if not databases:
                logger.warning("No databases found or all databases were excluded.")
                return

            # Get tables for each database
            tables_by_db = {}
            for db in databases:
                db_name = db['Name']
                tables = self.get_tables(db_name)
                # Filter excluded tables
                tables = [table for table in tables if table['Name'] not in self.exclude_tables]
                tables_by_db[db_name] = tables

            # Create output directory if it doesn't exist
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to markdown file
            with self.output_file.open("w", encoding="utf-8") as md_file:
                # Write header and TOC
                if not self.no_header:
                    md_file.write(self.create_header())
                    md_file.write(self.generate_toc(databases, tables_by_db))
                    md_file.write("\n---\n\n")

                # Write database and table information
                for db in databases:
                    db_name = db['Name']
                    md_file.write(f"## {db_name}\n\n")

                    # Add database description if available
                    if 'Description' in db and db['Description']:
                        md_file.write(f"{db['Description']}\n\n")

                    # Add database metadata
                    md_file.write("**Metadata:**\n\n")
                    md_file.write(f"- **Created:** {db.get('CreateTime', 'Unknown')}\n")

                    if 'Parameters' in db and db['Parameters']:
                        md_file.write("\n**Parameters:**\n\n")
                        for key, value in db['Parameters'].items():
                            md_file.write(f"- **{key}:** {value}\n")

                    md_file.write("\n")

                    # Write tables information
                    if db_name in tables_by_db and tables_by_db[db_name]:
                        for table in tables_by_db[db_name]:
                            table_schema = self.generate_table_schema(table)
                            md_file.write(table_schema)
                            md_file.write("\n---\n\n")
                    else:
                        md_file.write("*No tables found in this database.*\n\n---\n\n")

            logger.info(f"Glue documentation created: {self.output_file}")

        except Exception as e:
            logger.error(f"Error creating documentation: {e}")
            raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create a Markdown documentation of AWS Glue databases and tables."
    )
    parser.add_argument(
        "-o", "--output",
        default="glue_documentation.md",
        help="Name of the output Markdown file."
    )
    parser.add_argument(
        "-r", "--region",
        help="AWS region to use."
    )
    parser.add_argument(
        "-p", "--profile",
        help="AWS profile to use."
    )
    parser.add_argument(
        "-d", "--database",
        help="Document only the specified database."
    )
    parser.add_argument(
        "--exclude-databases",
        nargs="*",
        default=[],
        help="List of database names to exclude."
    )
    parser.add_argument(
        "--exclude-tables",
        nargs="*",
        default=[],
        help="List of table names to exclude."
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not include the header in the documentation."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    generator = GlueDocumentationGenerator(
        output_file=args.output,
        region=args.region,
        profile=args.profile,
        database_name=args.database,
        exclude_databases=set(args.exclude_databases),
        exclude_tables=set(args.exclude_tables),
        no_header=args.no_header
    )

    generator.create_documentation()


if __name__ == "__main__":
    main()
