#!/bin/bash
# Wrapper for cmake that filters out the sbt-jni version parsing warning
# Redirect stderr to stdout, filter, then redirect back to stderr
exec 3>&1
/usr/bin/cmake "$@" 2>&1 >&3 3>&- | grep -v -E "(CMake Warning:|Ignoring extra path from command line|/build/[0-9]+\")" >&2
exit ${PIPESTATUS[0]}
