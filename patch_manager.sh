#!/bin/bash

# Usage:
#   ./patch_manager.sh update
#   ./patch_manager.sh apply
#
# Description:
#   This script either updates/creates patches for submodules
#   or applies patches to submodules, depending on user input.

COMMAND="$1"

# Display usage if no argument is provided
if [ -z "$COMMAND" ]; then
  echo "Usage: $0 [update|apply]"
  exit 1
fi

confirm_action() {
  read -p "To confirm, please retype '$1': " CONFIRM
  if [ "$CONFIRM" != "$1" ]; then
    echo "Action $1 cancelled."
    exit 0
  fi
}

case "$COMMAND" in
  update)
    confirm_action "update"
    echo "Updating/creating patches..."
    # Create a hidden directory for patches if it doesn't exist
    mkdir -p "$(pwd)/.patches"

    # Iterate over each submodule under the external/ directory
    git submodule foreach --recursive '
      if [ "${sm_path#external/}" != "$sm_path" ]; then
        # Get the submodule name
        submodule_name=$(basename "$(git rev-parse --show-toplevel)")
        
        # Generate the patch file name using only the submodule name
        patch_file="$(git rev-parse --show-superproject-working-tree)/.patches/${submodule_name}.patch"
        
        # Stage all changes (including new files)
        git add -A
        
        # Dump staged changes into a patch file
        git diff --cached > "$patch_file"
        
        # Remove empty patch files
        if [ ! -s "$patch_file" ]; then
          rm -f "$patch_file"
        else
          echo "Patch created: $patch_file"
        fi
    
        # Undo the git add
        git reset
      fi
    '
    
    echo "Patches have been created in the .patches directory."
    ;;
    
  apply)
    confirm_action "apply"
    echo "Applying patches..."

    # This section applies each patch file in the `.patches` directory
    # to the matching submodule directory based on the submodule name.

    for patch_file in .patches/*.patch; do
      [ -e "$patch_file" ] || break

      # Extract the submodule name from the patch file name
      submodule_name=$(basename "$patch_file" .patch)

      # Check if a directory with the submodule name exists under external/
      if [ -d "external/$submodule_name" ]; then
        echo "Reverting submodule: $submodule_name to original state"
        cd "external/$submodule_name" || exit 1
        git checkout .
        echo "Applying patch to submodule: $submodule_name"
        git apply "../../.patches/$submodule_name.patch"
        cd - || exit 1
      else
        echo "Submodule directory external/$submodule_name not found. Skipping..."
      fi
    done

    echo "All applicable patches have been applied."

    echo "Reinstalling Packages."
    set -e  # stop if anything fails

    # Install external submodules in editable mode
    pip install -e ./external/robosuite
    pip install -e ./external/robomimic
    pip install -e ./external/mimicgen
    pip install -e ./external/equidiff

    # replace mujoco version to 2.3.2
    pip install mujoco==2.3.2

    # Install your own package
    pip install -e mirrorduo
    ;;

  *)
    echo "Invalid command: $COMMAND"
    echo "Usage: $0 [update|apply]"
    exit 1
    ;;
esac
