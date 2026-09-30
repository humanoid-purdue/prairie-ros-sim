# Dev environment for prairie-ros-sim.
# ROS 2 Jazzy (LTS) on Ubuntu 24.04, Gazebo Harmonic, ros2_control, and the
# Python libraries our nodes import, pinned from uv.lock.
#
# The repository is bind-mounted at /ws (see compose.yaml). Nothing from src/ is
# copied in, so edits on the host are visible in the container immediately and
# code changes never force an image rebuild.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- ROS / Gazebo packages -------------------------------------------------
# Taken from the package.xml files plus what the launch files start.
# If you add a <depend> to a package.xml, add its apt package here too.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-pip \
      python3-dev \
      python3-pytest \
      ros-jazzy-ros-gz \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-control-msgs \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-rviz2 \
      ros-jazzy-joy \
      ros-jazzy-ament-lint-auto \
      ros-jazzy-ament-lint-common \
      libgl1 libegl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# --- Python libraries --------------------------------------------------------
# ROS nodes run on the system Python (3.12), not a venv, so the locked versions
# are installed there. setuptools/packaging are left to apt: newer pip versions
# of them break colcon's ament_python builds.
COPY pyproject.toml uv.lock /tmp/py/
RUN pip3 install --no-cache-dir --break-system-packages uv \
    && cd /tmp/py \
    && uv export --frozen --format requirements-txt --no-hashes --no-emit-project \
         --no-dev --group policy --group teleop \
       | grep -vE '^(setuptools|packaging)==' > requirements.txt \
    && pip3 install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt \
    && rm -rf /tmp/py

# --- Browser desktop (noVNC) -------------------------------------------------
# Xvnc is the X server; OpenGL is Mesa's software renderer (llvmpipe), since
# Docker Desktop on macOS/Windows has no GPU passthrough. See docker/start-vnc.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tigervnc-standalone-server \
      novnc websockify \
      fluxbox xterm \
      libgl1-mesa-dri mesa-utils \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    RCUTILS_COLORIZED_OUTPUT=1

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/start-vnc.sh /usr/local/bin/start-vnc
# `docker compose exec` shells and desktop xterms skip the entrypoint, so source
# ROS from .bashrc too.
RUN chmod +x /entrypoint.sh /usr/local/bin/start-vnc \
    && printf '%s\n' \
         'source /opt/ros/jazzy/setup.bash' \
         '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' \
         >> /root/.bashrc

WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]