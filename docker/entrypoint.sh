#!/bin/bash
# Source ROS, then this workspace's overlay if it has been built.
set -e
source /opt/ros/jazzy/setup.bash
if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
fi
exec "$@"