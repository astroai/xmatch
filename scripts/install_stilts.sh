#!/bin/bash

# Script to create a 'stilts' command that uses the JAR from a local TOPCAT installation.

# Configuration
INSTALL_DIR="$HOME/bin"
STILTS_SCRIPT_PATH="$INSTALL_DIR/stilts"
TOPCAT_APP_NAME="TOPCAT.app"
SEARCH_DIRS=("/Applications") # Add other likely locations if needed

# --- Function to find TOPCAT JAR ---
find_topcat_jar() {
    local search_dir
    local topcat_path
    local jar_path_full
    local jar_path_lite

    echo "Searching for $TOPCAT_APP_NAME in ${SEARCH_DIRS[@]}..."

    for search_dir in "${SEARCH_DIRS[@]}"; do
        topcat_path=$(find "$search_dir" -maxdepth 2 -name "$TOPCAT_APP_NAME" -type d -print -quit)
        if [ -n "$topcat_path" ]; then
            echo "Found TOPCAT at: $topcat_path"

            # Look for topcat-full.jar first, then topcat-lite.jar
            jar_path_full=$(find "$topcat_path/Contents/Resources/" -name "topcat-extra.jar" -print -quit)
            if [ -f "$jar_path_full" ]; then
                echo "Using JAR: $jar_path_full"
                echo "$jar_path_full" # Return the path
                return 0
            fi

            jar_path_lite=$(find "$topcat_path/Contents/Resources/" -name "topcat-lite.jar" -print -quit)
             if [ -f "$jar_path_lite" ]; then
                echo "Using JAR (lite): $jar_path_lite"
                echo "$jar_path_lite" # Return the path
                return 0
            fi

            echo "Error: Found TOPCAT app, but could not find topcat-full.jar or topcat-lite.jar inside."
            echo "Looked in: $topcat_path/Contents/Resources/"
            return 1
        fi
    done

    echo "Error: Could not find $TOPCAT_APP_NAME in ${SEARCH_DIRS[@]}."
    echo "Please ensure TOPCAT is installed in a standard location or adjust SEARCH_DIRS in the script."
    return 1
}
# --- End Function ---


# Find the TOPCAT JAR path
TOPCAT_JAR_PATH=$(find_topcat_jar)
if [ $? -ne 0 ]; then
    exit 1
fi
if [ -z "$TOPCAT_JAR_PATH" ]; then
    echo "Error: Failed to get TOPCAT JAR path."
    exit 1
fi

# Ensure installation directory exists
mkdir -p "$INSTALL_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create installation directory $INSTALL_DIR"
  exit 1
fi


# Create wrapper script
echo "Creating wrapper script at $STILTS_SCRIPT_PATH..."
# Use the located TOPCAT_JAR_PATH in the wrapper script
cat << EOF > "$STILTS_SCRIPT_PATH"
#!/bin/bash
# Wrapper script for STILTS, using the TOPCAT JAR
# Executes the TOPCAT JAR file, passing all arguments

# Path to the TOPCAT JAR determined during installation
JAR_PATH="$TOPCAT_JAR_PATH"

# Check if Java is available
if ! command -v java &> /dev/null; then
    echo "Error: Java command not found. Please install Java."
    exit 1
fi

# Check if JAR file exists
if [ ! -f "\$JAR_PATH" ]; then
    echo "Error: TOPCAT JAR file not found at \$JAR_PATH"
    echo "The TOPCAT application may have been moved or deleted."
    exit 1
fi

# Execute STILTS functionality via TOPCAT JAR
# Add -stilts flag to invoke STILTS mode
echo "Running command via TOPCAT JAR: java -jar \"\$JAR_PATH\" -stilts \$@" >&2 # Debug output to stderr
exec java -jar "\$JAR_PATH" -stilts "\$@"
EOF

# Make wrapper script executable
chmod +x "$STILTS_SCRIPT_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to make wrapper script executable."
  # No JAR to clean up in this version
  exit 1
fi
echo "Created executable wrapper script at $STILTS_SCRIPT_PATH"

# Check if INSTALL_DIR is in PATH and advise if not
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo ""
  echo "Warning: Installation directory $INSTALL_DIR is not in your PATH."
  echo "To run the 'stilts' command easily, add the following line to your ~/.zshrc file:"
  echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
  echo "Then, restart your terminal or run 'source ~/.zshrc'."
else
    echo ""
    echo "$INSTALL_DIR is already in your PATH."
fi

echo ""
echo "STILTS wrapper script using TOPCAT JAR created successfully!"
echo "You can now try running: stilts -help"
