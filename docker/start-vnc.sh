#!/bin/bash
# Headless X desktop reachable from a browser, for running Gazebo/RViz in the
# container without an X server on the host.
#
#   Xvnc (X server + VNC on :5901) -> websockify/noVNC on :6080 -> browser
#
# Open http://localhost:6080/vnc.html once this is running. Any GUI started with
# DISPLAY=:1 (the default in the `gui` compose service) shows up there.
set -e

: "${DISPLAY:=:1}"
: "${VNC_RESOLUTION:=1920x1080}"
: "${NOVNC_PORT:=6080}"
export DISPLAY
display_num="${DISPLAY#:}"
vnc_port=$((5900 + display_num))

# Leftovers from a previous run of this container would stop Xvnc starting.
rm -f "/tmp/.X${display_num}-lock" "/tmp/.X11-unix/X${display_num}"

# No VNC password: the port is only published on the host's localhost.
Xvnc "$DISPLAY" \
  -geometry "$VNC_RESOLUTION" -depth 24 \
  -SecurityTypes None -localhost yes -rfbport "$vnc_port" \
  -AlwaysShared -AcceptSetDesktopSize \
  >/tmp/xvnc.log 2>&1 &

for _ in $(seq 50); do
  [ -S "/tmp/.X11-unix/X${display_num}" ] && break
  sleep 0.1
done

fluxbox >/tmp/fluxbox.log 2>&1 &
xterm -geometry 120x35+20+20 >/dev/null 2>&1 &

echo "noVNC: http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=remote"
exec websockify --web /usr/share/novnc "$NOVNC_PORT" "localhost:${vnc_port}"
