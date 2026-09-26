import time
import vgamepad as vg

from config import START_DELAY, START_HOLD_TIME


class GameController:

    def __init__(self):
        """
        Creates the virtual Xbox 360 controller.
        """

        self.gamepad = vg.VX360Gamepad()

        # Used for the automatic START button
        self.start_time = time.time()
        self.start_pressed = False
        self.start_released = False


    def handle_auto_start(self):
        """
        Automatically presses the Xbox START button
        after START_DELAY seconds.
        """

        elapsed = time.time() - self.start_time

        # Press START
        if elapsed >= START_DELAY and not self.start_pressed:

            self.gamepad.press_button(
                button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START
            )

            self.gamepad.update()

            self.start_pressed = True

        # Release START
        if (
            elapsed >= START_DELAY + START_HOLD_TIME
            and self.start_pressed
            and not self.start_released
        ):

            self.gamepad.release_button(
                button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START
            )

            self.gamepad.update()

            self.start_released = True


    def steer(self, steering_value):
        """
        Controls the Xbox left joystick.

        steering_value should be between:
        -1.0 = full left
         0.0 = center
        +1.0 = full right
        """

        self.gamepad.left_joystick_float(
            x_value_float=steering_value,
            y_value_float=0.0
        )


    def accelerate(self):
        """
        Full acceleration using the right trigger.
        """

        self.gamepad.right_trigger_float(
            value_float=1.0
        )

        self.gamepad.left_trigger_float(
            value_float=0.0
        )


    def brake(self):
        """
        Full brake/reverse using the left trigger.
        """

        self.gamepad.right_trigger_float(
            value_float=0.0
        )

        self.gamepad.left_trigger_float(
            value_float=1.0
        )


    def coast(self):
        """
        Releases both triggers.
        """

        self.gamepad.right_trigger_float(
            value_float=0.0
        )

        self.gamepad.left_trigger_float(
            value_float=0.0
        )


    def update(self):
        """
        Sends the current controller state to Windows.
        """

        self.gamepad.update()


    def reset(self):
        """
        Returns the controller to its neutral state.
        """

        self.gamepad.reset()
        self.gamepad.update()