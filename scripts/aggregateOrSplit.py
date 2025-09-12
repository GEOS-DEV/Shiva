import os
import re
import argparse

class HeaderFileManager:
    def __init__(self):
        self.dependencies = {}  # Dictionary of header files and their dependencies
        self.dependents = {}   # Dictionary of header files and their dependents
        self.included = set()  # Tracks already processed files
        self.included_list = []
        self.config_file = "ShivaConfig.hpp"

    def create_dependency_graph(self, header, include_paths=None):
        """
        Creates a dependency graph for the given header file, resolving dependencies from:
        1. The same directory as the file being parsed.
        2. A relative path specified in the #include directive.
        3. A list of include paths (e.g., provided in the compile line).
        """
        if include_paths is None:
            include_paths = []

        header = os.path.abspath(header)  # Normalize here

        if header in self.dependencies:
            return  # Already processed

        self.dependencies[header] = set()
        base_path = os.path.dirname(header)  # Base directory of the current header

        try:
            with open(header, 'r') as file:
                for line in file:
                    include_match = re.match(r'#include\s+"([^"]+)"', line)
                    if include_match:
                        included_file = include_match.group(1)

                        if included_file != self.config_file:
                            resolved_path = self.resolve_path( included_file, base_path, include_paths)

                            if resolved_path:
                                resolved_path = os.path.abspath(resolved_path)
                                self.dependencies[header].add(resolved_path)

                                if os.path.exists(resolved_path):
                                    # Recursively process the resolved file
                                    self.create_dependency_graph(resolved_path, include_paths)
                                else:
                                    raise FileNotFoundError(f"Dependency not found: {resolved_path}")
                            else:
                                raise FileNotFoundError(
                                    f"Unable to resolve dependency: {included_file}")
        except Exception as e:
            print(f"Error processing {header}: {e}")

        # Update dependents
        for dependency in self.dependencies[header]:
            if dependency not in self.dependents:
                self.dependents[dependency] = set()
            self.dependents[dependency].add(header)

    def resolve_path(self, included_file, base_path, include_paths):
        """
        Resolves the path of an included file using the following order:
        1. Check if the file exists in the same directory as the current file.
        2. Check if the file exists using a relative path.
        3. Check if the file exists in any of the provided include paths.
        """
        # 1. Check in the same directory
        same_dir_path = os.path.join(base_path, included_file)
        if os.path.exists(same_dir_path):
            return os.path.normpath(same_dir_path)

        # 2. Check using the relative path
        relative_path = os.path.normpath(os.path.join(base_path, included_file))
        if os.path.exists(relative_path):
            return relative_path

        # 3. Check in the include paths
        for include_path in include_paths:
            candidate_path = os.path.normpath(os.path.join(include_path, included_file))
            if os.path.exists(candidate_path):
                return candidate_path

        return None  # Return None if no resolution was possible


    def generate_header_list(self):
        remaining_dependencies = self.dependencies.copy()
        size_of_remaining_dependencies = len(remaining_dependencies)
        unique_files = set()  # Track unique files by absolute path

        while size_of_remaining_dependencies > 0:
            local_included = []

            for key in remaining_dependencies:
                if len(remaining_dependencies[key]) == 0:
                    abs_key = os.path.abspath(key)
                    if abs_key not in unique_files:
                        self.included_list.append(abs_key)
                        unique_files.add(abs_key)
                    local_included.append(key)

            for included_key in local_included:
                del remaining_dependencies[included_key]

                for key in remaining_dependencies:
                    if included_key in remaining_dependencies[key]:
                        remaining_dependencies[key].remove(included_key)

            size_of_remaining_dependencies = len(remaining_dependencies)

    def aggregate_headers(self, headers, output_file, include_paths=None):
        """
        Aggregates header files into a single file, resolving dependencies.
        """
        def process_header(header_path, output):
            """
            Processes a single header file, commenting out includes and pragmas.
            """
            header_path = os.path.abspath(header_path)
            if header_path in self.included:
                return  # Avoid duplicate processing
            self.included.add(header_path)

            with open(header_path, 'r') as file:
                output.write(f"\n\n// ===== Start of {header_path} =====\n")
                for line in file:
                    include_match = re.match(r'#include\s+"([^"]+)"', line)
                    pragma_once_match = re.match(r'#pragma\s+once', line, re.IGNORECASE)
                    if include_match:
                        # Comment out include statements
                        output.write(f"// {line.strip()}\n")
                    elif pragma_once_match:
                        # Comment out #pragma once
                        output.write(f"// {line.strip()}\n")
                    else:
                        # Write all other lines as is
                        output.write(line)
                output.write(f"// ===== End of {header_path} =====\n")

        with open(output_file, 'w') as output:
            for header in headers:
                header = os.path.abspath(header)
                self.create_dependency_graph(header, include_paths)
            
            for header in self.dependencies:
                print( f"Header: {header} -> {self.dependencies[header]}\n")
            print( "\n\n")

            self.generate_header_list()
            print( f"Header List: {self.included_list}\n")

            for header in self.included_list:
                process_header(header, output)

    def split_aggregated_file(self, aggregated_file, output_dir):
        """
        Splits an aggregated file back into individual header files.
        """
        os.makedirs(output_dir, exist_ok=True)
        current_file = None
        output = None

        with open(aggregated_file, 'r') as agg_file:
            for line in agg_file:
                start_match = re.match(r'// ===== Start of (.+) =====', line)
                end_match = re.match(r'// ===== End of (.+) =====', line)

                if start_match:
                    if output:
                        output.close()
                    current_file = start_match.group(1)
                    output = open(os.path.join(output_dir, current_file), 'w')
                elif end_match:
                    if output:
                        output.close()
                        output = None
                elif output:
                    output.write(line)

        if output:
            output.close()

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Aggregate or split header files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Aggregate command
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate header files into a single file.")
    aggregate_parser.add_argument("headers", nargs='+', help="List of header files to aggregate.")
    aggregate_parser.add_argument("output", help="Output file for the aggregated headers.")
    aggregate_parser.add_argument("--include-paths", nargs='*', default=[os.getcwd()], help="List of include paths to resolve dependencies. Defaults to the current directory.")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split an aggregated file into individual header files.")
    split_parser.add_argument("aggregated_file", help="Aggregated header file to split.")
    split_parser.add_argument("output_dir", help="Directory to store the split header files.")

    args = parser.parse_args()
    manager = HeaderFileManager()

    if args.command == "aggregate":
        manager.aggregate_headers(args.headers, args.output, args.include_paths)
    elif args.command == "split":
        manager.split_aggregated_file(args.aggregated_file, args.output_dir)

if __name__ == "__main__":
    main()
