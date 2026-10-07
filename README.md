# prairie-ros-sim

ROS 2 simulation and control workspace for the Nemo robot, using ROS 2 Jazzy,
Gazebo Harmonic, and Python 3.12. The main simulation launch runs Nemo6 with
keyboard or joystick control; real hardware nodes are disabled by default.

## Quick start: Docker with a browser desktop

Install Docker with Docker Compose support (Docker Desktop on macOS or Windows),
start Docker, and open a terminal in the repository root. The image includes ROS,
Gazebo, and the Python dependencies from `uv.lock`.

### 1. Build and start the environment

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml up -d gui
docker compose -f docker/compose.yaml exec gui bash
```

The first image build can take several minutes. The repository is mounted at
`/ws`, so edits on your host are available inside the container.

### 2. Build the ROS workspace and launch

Run these commands **inside the container**:

```bash
cd /ws
colcon build --symlink-install
source install/setup.bash
ros2 launch prairie_control master_gzlink.launch.py use_joy:=false use_rviz:=false
```

Open [the browser desktop](http://localhost:6080/vnc.html?autoconnect=1&resize=remote)
to see Gazebo and the keyboard input window. The robot spawns after a 15-second
delay. Keep the keyboard input window focused while driving:

| Key | Action |
| --- | --- |
| `1` / `2` | Stand / walk |
| `W` / `S` | Forward / backward |
| `A` / `D` | Left / right |
| `Q` / `E` | Turn left / right |
| `0` | Emergency stop |

Releasing a movement key stops that command; losing keyboard focus clears all
movement commands. See [the controls guide](docs/master_gzlink_controls.md) for
speed settings, joystick mappings, and hardware modes.

### 3. Stop the environment

Press `Ctrl+C` in the launch terminal, then run this **on the host**:

```bash
docker compose -f docker/compose.yaml down
```

## Other launch options

Inside the built and sourced ROS workspace:

```bash
# Keyboard control with RViz enabled.
ros2 launch prairie_control master_gzlink.launch.py use_joy:=false

# Joystick control with RViz (requires a joystick accessible to ROS).
ros2 launch prairie_control master_gzlink.launch.py

# Gazebo-only Nemo6 scene, without the high-level control stack.
ros2 launch gz_sim gz_only_nemo6.launch.py

# List the main launch arguments.
ros2 launch prairie_control master_gzlink.launch.py --show-args
```

Joystick input defaults to enabled; keyboard input is selected with
`use_joy:=false`. RViz defaults to enabled. Real motor and IMU nodes start only
with `use_hardware:=true`; consult the controls guide before using hardware.

For a Linux host with an X server, the `dev` service can use your local display:
run `xhost +local:docker`, uncomment the X11 volume and `DISPLAY` entries in
[docker/compose.yaml](docker/compose.yaml), then start a shell with:

```bash
docker compose -f docker/compose.yaml run --rm dev
```

Build, source, and launch as above. The browser desktop requires no host X11
configuration, including on Windows.

## Development and checks

For Python tests on the host, install [uv](https://docs.astral.sh/uv/) and run from
the repository root:

```bash
uv sync --locked
uv run ruff check --select E9,F63,F7,F82 .
uv run pytest -q
```

These match the Python checks in CI. The host virtual environment is for Python
development; ROS nodes in Docker use the container's system Python.

After ROS package changes, rebuild with `colcon build --symlink-install` and
source `install/setup.bash` in each terminal. Rebuild the Docker image when
changing its dependencies or startup scripts. When adding ROS dependencies,
update both the package's `package.xml` and [docker/Dockerfile](docker/Dockerfile).

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution guidelines.
