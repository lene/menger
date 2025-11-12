#!/bin/bash

# Automated window resize test for OptiX sphere rendering
# This script tests the resize behavior automatically

echo "Starting OptiX window resize test..."
echo "This test requires xdotool to be installed"

# Start the application with specific parameters for better visibility
echo "Launching application with green sphere..."
ENABLE_OPTIX_JNI=true sbt "run --optix --sponge-type sphere --radius 0.5 --color 00ff00" &
APP_PID=$!

# Wait for window to appear
echo "Waiting for application window to appear..."
sleep 10

# Find window ID using xdotool
WINDOW_ID=$(xdotool search --name "Menger" | head -1)

if [ -z "$WINDOW_ID" ]; then
    echo "ERROR: Could not find application window"
    echo "Make sure the application launched successfully"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Found window ID: $WINDOW_ID"
echo ""

# Function to resize and wait
resize_window() {
    local width=$1
    local height=$2
    local description=$3

    echo "Test: $description"
    echo "  Resizing to ${width}x${height}..."
    xdotool windowsize $WINDOW_ID $width $height
    sleep 3
    echo "  Done"
    echo ""
}

# Test sequence based on the specification
echo "=== Starting test sequence ==="
echo ""

echo "Initial state: 800x600 (baseline)"
sleep 3

resize_window 1200 600 "Increase width to 1200x600 (expect: sphere scales up, stays circular)"

resize_window 600 600 "Decrease width to 600x600 (expect: sphere scales down, stays circular)"

resize_window 600 900 "Increase height to 600x900 (expect: sphere same size, stays circular)"

resize_window 800 600 "Return to original 800x600"

resize_window 1600 600 "Double width to 1600x600 (expect: sphere scales 2x, stays circular)"

resize_window 800 1200 "Double height to 800x1200 (expect: sphere same size, stays circular)"

resize_window 800 600 "Final return to 800x600"

echo "=== Test sequence complete ==="
echo ""

echo "Keeping application open for 5 more seconds for manual inspection..."
sleep 5

echo "Closing application..."
kill $APP_PID 2>/dev/null

echo "Test complete!"
echo ""
echo "Expected results per specification:"
echo "  - Width changes: sphere should scale proportionally but stay circular"
echo "  - Height changes: sphere should stay same size and circular"
echo "  - No distortion (elliptical shapes) should occur"
echo "  - No black borders should appear"