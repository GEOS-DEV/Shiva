#!/usr/bin/env python
# Python wrapper script for generating the correct cmake line with the options specified by the user.
#
# Please keep parser option names as close to possible as the names of the cmake options they are wrapping.

import sys
import os
import subprocess
import argparse
import platform
import shutil

def extract_cmake_location(file_path):
    print("Extracting cmake entry from host config file ", file_path)
    if os.path.exists(file_path):
        cmake_line_prefix = "# cmake executable path: "
        file_handle = open(file_path, "r")
        content = file_handle.readlines()
        for line in content:
            if line.startswith(cmake_line_prefix):
                return line.split(" ")[4].strip()
        print("Could not find a cmake entry in host config file.")
    return None



parser = argparse.ArgumentParser(description="Configure cmake build. Unrecognized arguments are passed on to CMake.")

parser.add_argument("-bp",
                    "--buildpath",
                    type=str,
                    default="",
                    help="specify path for build directory.  If not specified, will create in current directory.")

parser.add_argument("-ip",
                    "--installpath", 
                    type=str, default="",
                    help="specify path for installation directory.  If not specified, will create in current directory.")

parser.add_argument("-bt",
                    "--buildtype",
                    type=str,
                    choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
                    default="Debug",
                    help="build type.")

parser.add_argument("-e",
                    "--eclipse",
                    action='store_true',
                    help="create an eclipse project file.")

parser.add_argument("-x",
                    "--xcode",
                    action='store_true',
                    help="create an xcode project.")

parser.add_argument("-ecc",
                    "--exportcompilercommands",
                    action='store_true',
	                help="generate a compilation database.  Can be used by the clang tools such as clang-modernize.  Will create a file called 'compile_commands.json' in build directory.")

parser.add_argument("-hc",
                    "--hostconfig",
                    required=True,
                    type=str,
                    help="select a specific host-config file to initalize CMake's cache")

parser.add_argument("-n", "--ninja", action='store_true', help="Create a ninja project.")
parser.add_argument("-gvz", "--graphviz", action="store_true", help="Generate graphviz dependency graph")



args, unknown_args = parser.parse_known_args()
if unknown_args:
    print("[config-build]: Passing the following unknown arguments directly to cmake... %s" % unknown_args)

########################
# Find CMake Cache File
########################
platform_info = ""
scriptsdir = os.path.dirname( os.path.abspath(sys.argv[0]) )

cachefile = os.path.abspath(args.hostconfig)
platform_info = os.path.split(cachefile)[1]
if platform_info.endswith(".cmake"):
    platform_info = platform_info[:-6]
    
assert os.path.exists( cachefile ), "Could not find cmake cache file '%s'." % cachefile
print("Using host config file: '%s'." % cachefile)

#####################
# Setup Build Dir
#####################
if args.buildpath != "":
    # use explicit build path
    buildpath = args.buildpath
else:
    # use platform info & build type
    buildpath = "-".join(["build",platform_info,args.buildtype.lower()])

buildpath = os.path.abspath(buildpath)

if os.path.exists(buildpath):
#    sys.exit("Build directory '%s' already exists, exiting...")
     print("Build directory '%s' already exists.  Deleting..." % buildpath)
     shutil.rmtree(buildpath)

print("Creating build directory '%s'..." % buildpath)
os.makedirs(buildpath)

#####################
# Setup Install Dir
#####################
# For install directory, we will clean up old ones, but we don't need to create it, cmake will do that.
if args.installpath != "":
    installpath = os.path.abspath(args.installpath)
else:
    # use platform info & build type
    installpath = "-".join(["install",platform_info,args.buildtype.lower()])

installpath = os.path.abspath(installpath)

if os.path.exists(installpath):
#    sys.exit("Install directory '%s' already exists, exiting...")
     print("Install directory '%s' already exists, deleting..." % installpath)
     shutil.rmtree(installpath)

print("Creating install path '%s'..." % installpath)
os.makedirs(installpath)

############################
# Build CMake command line
############################

cmakeline = extract_cmake_location(cachefile)
cmakeline = "cmake"
assert cmakeline, "Host config file doesn't contain valid cmake location, value was %s" % cmakeline

# Add cache file option
cmakeline += " -C %s" % cachefile
# Add build type (opt or debug)
cmakeline += " -DCMAKE_BUILD_TYPE=" + args.buildtype
# Set install dir
cmakeline += " -DCMAKE_INSTALL_PREFIX=%s" % installpath

if args.exportcompilercommands:
    cmakeline += " -DCMAKE_EXPORT_COMPILE_COMMANDS=on"

if args.eclipse:
    cmakeline += ' -G "Eclipse CDT4 - Unix Makefiles"'

if args.xcode:
    cmakeline += ' -G Xcode'

if args.ninja:
    cmakeline += ' -GNinja'

if args.graphviz:
    cmakeline += " --graphviz=dependency.dot"
    dot_line = "dot -Tpng dependency.dot -o dependency.png"


if unknown_args:
    cmakeline += " " + " ".join( unknown_args )

cmakeline += " %s/.. " % scriptsdir

# Dump the cmake command to file for convenience
cmdfile = open("%s/cmake_cmd" % buildpath, "w")
cmdfile.write(cmakeline)
cmdfile.close()
import stat
st = os.stat("%s/cmake_cmd" % buildpath)
os.chmod("%s/cmake_cmd" % buildpath, st.st_mode | stat.S_IEXEC)

############################
# Run CMake
############################
print("Changing to build directory...")
os.chdir(buildpath)
print("Executing cmake line: '%s'\n" % cmakeline)

try:
    subprocess.call(cmakeline,shell=True)
    if args.graphviz:
        subprocess.call(dot_line, shell=True)
except:
    print("CMake failed.  See above output for details.")
    sys.exit(1)

