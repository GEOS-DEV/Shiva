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

    def create_dependency_graph(self, header):
        self.dependencies[header] = set()
        path = os.path.dirname(header)
        with open(header, 'r') as file:
            for line in file:
                include_match = re.match(r'#include\s+"([^"]+)"', line)
                if include_match:
                    included_file = include_match.group(1)

                    if( included_file != self.config_file):

                        if( not os.path.dirname(included_file) ):
                            included_file = os.path.join(path, included_file)

                        self.dependencies[header].add(included_file)

                        if os.path.exists(included_file):
                            self.create_dependency_graph(included_file)

            for dependency in self.dependencies[header]:
                if dependency not in self.dependents:
                    self.dependents[dependency] = set()
                self.dependents[dependency].add(header)
    
    def generate_header_list(self):

        remainingDependencies = self.dependencies.copy()

        sizeOfRemainingDependencies = len(remainingDependencies)
        # print( "sizeOfRemainingDependencies: " + str(sizeOfRemainingDependencies) )

        while sizeOfRemainingDependencies > 0:
            localIncluded = []

            # print( "\nRemaining Dependencies: "  )
            # max_key_length = max(len(key) for key in remainingDependencies)
            # for key, value in remainingDependencies.items():
            #     print(f"{key.ljust(max_key_length)}: {value}")

            for key in remainingDependencies:
                if len( remainingDependencies[key] )==0:
                    self.included_list.append(key)
                    localIncluded.append(key)

            for includedKey in localIncluded:
                
                del remainingDependencies[includedKey]

                for key in remainingDependencies:
                    if includedKey in remainingDependencies[key]:
                        remainingDependencies[key].remove(includedKey)



            # print( "\nlocalIncluded: " )
            # print( localIncluded )
            # print( self.included_list )

            sizeOfRemainingDependencies = len(remainingDependencies)
            # print( "\nsizeOfRemainingDependencies: " + str(sizeOfRemainingDependencies) )

        # print( "Included List: ")
        # for header in self.included_list:
        #     print( header)




    def aggregate_headers(self, headers, output_file):

        """
        Aggregates header files into a single file, resolving dependencies.
        """

        def process_header(header_path, output):
            """
            Processes a single header file, commenting out includes and pragmas.
            """
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
                self.create_dependency_graph(header)
                print( "Dependencies: ")
                for key in self.dependencies:
                    print( f"{key}: {self.dependencies[key]}")

            self.generate_header_list()

                # print( "Dependents: ")
                # print(self.dependents)

            for header in self.included_list:
                process_header(header, output)






def split_aggregated_file(self, aggregated_file, output_dir):
    """
    Splits an aggregated file back into individual header files.
    Removes comment markers added during aggregation.
    """
    os.makedirs(output_dir, exist_ok=True)
    current_file = None
    output = None

    with open(aggregated_file, 'r') as agg_file:
        for line in agg_file:
            start_match = re.match(r'// ===== Start of (.+) =====', line)
            end_match = re.match(r'// ===== End of (.+) =====', line)

            if start_match:
                # Start a new header file
                if output:
                    output.close()
                current_file = start_match.group(1)
                output_path = os.path.join(output_dir, current_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directories exist
                output = open(output_path, 'w')
            elif end_match:
                # End the current header file
                if output:
                    output.close()
                    output = None
            elif output:
                # Write the line to the current header file
                # Remove comment markers from #include and #pragma once lines
                uncommented_line = re.sub(r'^//\s*', '', line)  # Remove leading "// "
                output.write(uncommented_line)

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

    # Split command
    split_parser = subparsers.add_parser("split", help="Split an aggregated file into individual header files.")
    split_parser.add_argument("aggregated_file", help="Aggregated header file to split.")
    split_parser.add_argument("output_dir", help="Directory to store the split header files.")

    args = parser.parse_args()
    manager = HeaderFileManager()

    if args.command == "aggregate":
        manager.aggregate_headers(args.headers, args.output)
    elif args.command == "split":
        manager.split_aggregated_file(args.aggregated_file, args.output_dir)

if __name__ == "__main__":
    main()
