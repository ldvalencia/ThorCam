"""
Live Video Capture with Normalized Intensity and Colorbar (Matplotlib)

This version displays the camera feed using matplotlib with a colorbar
representing the normalized intensity, and updates the plot live.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

# Optional: Windows DLL path setup
try:
    from windows_setup import configure_path
    configure_path()
except ImportError:
    pass

def normalize_image(image):
    """Normalize the image to 0-255 for visualization."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def get_camera_frame(camera):
    """Grab the latest frame from the camera."""
    frame = camera.get_pending_frame_or_null()
    if frame is not None:
        image = np.copy(frame.image_buffer)
        image_2d = image.reshape(camera.image_height_pixels, camera.image_width_pixels)
        return normalize_image(image_2d)
    else:
        return None

def main():
    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if not available_cameras:
            print("No cameras detected.")
            return

        with sdk.open_camera(available_cameras[0]) as camera:
            # Camera settings
            camera.exposure_time_us = 1000
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.image_poll_timeout_ms = 1000
            camera.frame_rate_control_value = 10
            camera.is_frame_rate_control_enabled = True

            camera.arm(2)
            camera.issue_software_trigger()

            # First frame to initialize the plot
            first_frame = None
            print("Starting live view with colorbar...")

            while first_frame is None:
                first_frame = get_camera_frame(camera)

            # Setup Matplotlib plot
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6))
            img_display = ax.imshow(first_frame, cmap='jet', vmin=0, vmax=255)
            cbar = fig.colorbar(img_display, ax=ax)
            cbar.set_label("Normalized Intensity", rotation=270, labelpad=15)
            ax.set_title("Live Normalized Intensity View")
            ax.axis('off')

            try:
                while True:
                    frame = get_camera_frame(camera)
                    if frame is not None:
                        img_display.set_data(frame)
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    else:
                        print("No frame received.")
            except KeyboardInterrupt:
                print("Interrupted by user.")
            finally:
                camera.disarm()
                print("Camera disarmed. Exiting...")
                plt.ioff()
                plt.close()

if __name__ == "__main__":
    main()
