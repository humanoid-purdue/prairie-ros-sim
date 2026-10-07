# Contributing

- Follow the [README](README.md) to set up and run the ROS/Gazebo environment.
- Create a branch and keep changes focused. Follow the existing code style and update documentation or tests when needed.
- For Python changes, install dependencies with `uv sync`, then run `uv run pytest` and `uv run ruff check .`.
- For ROS changes, run `colcon build --symlink-install` in the development container and verify the affected simulation or node.
- Open a pull request describing the change and how you tested it.

**When creating branches, use these standard prefixes**

| Prefix   | For                          | Example                         |
|----------|------------------------------|---------------------------------|
| `feat/`  | new capability               | `feat/headless-launch-arg`      |
| `fix/`   | bug fix                      | `fix/hardcoded-datadump-path`   |
| `docs/`  | README and docs only         | `docs/getting-started`          |
| `test/`  | adding tests                 | `test/math-utils`               |
| `ci/`    | workflows, Docker for CI     | `ci/tier1-colcon-build`         |
| `chore/` | cleanup, dependency bumps    | `chore/ruff-autofix`            |

For bug reports, include reproduction steps, expected and actual behavior, and relevant logs or environment details.

