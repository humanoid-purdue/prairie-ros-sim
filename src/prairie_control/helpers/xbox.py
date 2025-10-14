import pygame
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class StickState:
    x: float = 0.0
    y: float = 0.0

class XboxController:
    """Simple Xbox (USB) gamepad reader using pygame.

    Features:
      - Left & right stick (x,y) values in [-1,1] after deadzone filtering.
      - Button release mapping: A,B,X,Y,LB,RB -> integer states 0..5.
      - Non-blocking update(); poll periodically (e.g. in your main loop).
    """

    BUTTON_STATE_MAP = {  # pygame button index -> state int
        0: 0,  # A
        1: 1,  # B
        3: 2,  # X
        4: 3,  # Y
        6: 4,  # LB
        7: 5,  # RB
    }

    def __init__(self, joystick_id: int = 0, deadzone: float = 0.1):
        self.deadzone = float(deadzone)
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        if pygame.joystick.get_count() <= joystick_id:
            raise RuntimeError(f"No joystick at index {joystick_id} (count={pygame.joystick.get_count()})")
        self.js = pygame.joystick.Joystick(joystick_id)
        self.js.init()
        self.left = StickState()
        self.right = StickState()
        self.state: Optional[int] = 0  # Set on button release
        self._axis_map = self._detect_axis_layout()

    def _detect_axis_layout(self):
        axes = self.js.get_numaxes()
        # Common layouts:
        #  - 6 axes: 0=Lx,1=Ly,2=LT,3=Rx,4=Ry,5=RT (Xbox One)
        #  - 4 axes: 0=Lx,1=Ly,2=Rx,3=Ry (older / SDL config)
        if axes >= 5:
            return {"lx": 0, "ly": 1, "rx": 2, "ry": 3}
        elif axes >= 4:
            return {"lx": 0, "ly": 1, "rx": 2, "ry": 3}
        else:  # Fallback: duplicate left
            return {"lx": 0, "ly": 1, "rx": 0, "ry": 1}

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else float(max(-1.0, min(1.0, v)))

    def update(self):
        """Poll pygame events and refresh stick & state values."""
        # Process only relevant events to avoid queue growth
        for event in pygame.event.get([pygame.JOYAXISMOTION, pygame.JOYBUTTONUP]):
            if event.type == pygame.JOYBUTTONUP and event.joy == self.js.get_id():
                if event.button in self.BUTTON_STATE_MAP:
                    self.state = self.BUTTON_STATE_MAP[event.button]
        # Read axes (continuous)
        self.left.x = self._apply_deadzone(self.js.get_axis(self._axis_map["lx"]))
        self.left.y = self._apply_deadzone(-self.js.get_axis(self._axis_map["ly"]))  # invert Y for typical up positive
        self.right.x = self._apply_deadzone(self.js.get_axis(self._axis_map["rx"]))
        self.right.y = self._apply_deadzone(-self.js.get_axis(self._axis_map["ry"]))

    @property
    def left_stick(self) -> Tuple[float, float]:
        return (self.left.x, self.left.y)

    @property
    def right_stick(self) -> Tuple[float, float]:
        return (self.right.x, self.right.y)

    def get_state(self) -> Optional[int]:
        return self.state

    def close(self):  # optional cleanup
        try:
            self.js.quit()
        except Exception:
            pass

if __name__ == "__main__":  # simple demo
    import time
    ctrl = XboxController()
    print("Press A,B,X,Y,LB,RB to set state 0..5. Ctrl+C to exit.")
    try:
        while True:
            ctrl.update()
            print(f"L:{ctrl.left_stick} R:{ctrl.right_stick} state:{ctrl.get_state()}", end='\r')
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExiting")
    finally:
        ctrl.close()