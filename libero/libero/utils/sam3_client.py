"""
SAM3 ZMQ Client for real-time object segmentation.

This module provides a client for communicating with a SAM3 server
via ZMQ to obtain segmentation masks for objects in images.

Adapted for single-camera use from multi-camera streaming implementation.
"""

import os
import shutil
import cv2
import zmq
import msgpack
import numpy as np
from typing import Optional, Tuple


# Flag to control debug frame saving (set to False to reduce I/O overhead)
SAVE_FRAMES = True


class SAM3StreamClient:
    """Client for streaming frames to SAM3 and receiving segmented frames via ZMQ.

    Simplified for single-camera use (agentview only).
    """

    def __init__(
        self,
        send_endpoint: str = "tcp://localhost:5555",
        recv_endpoint: str = "tcp://localhost:5556",
        target_size: tuple[int, int] = (256, 256),  # (width, height) expected by SAM3
        original_size: tuple[int, int] = (256, 256),
        output_dir: str | None = "output_sam3",  # Directory to save debug frames
    ):
        """
        Initialize ZMQ sockets for SAM3 streaming.

        Args:
            send_endpoint: ZMQ endpoint to send frames to SAM3.
            recv_endpoint: ZMQ endpoint to receive segmented frames from SAM3.
            target_size: (width, height) to resize frames to before sending to SAM3.
            original_size: (width, height) of original frames.
            output_dir: Directory to save debug frames. None to disable saving.
        """
        self.target_size = target_size  # (width, height)
        self.original_size = original_size
        self.latest_segmented_frame: np.ndarray | None = None
        self._last_frame_shape: tuple[int, ...] | None = None
        self._frame_counter = 0
        self._output_dir = output_dir

        # Create output directory structure
        if output_dir and SAVE_FRAMES:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, "sent_images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "received_masks"), exist_ok=True)

        # Initialize ZMQ context and sockets
        self._context = zmq.Context()

        # PUSH socket to send frames to SAM3
        self._sender = self._context.socket(zmq.PUSH)
        self._sender.setsockopt(zmq.SNDHWM, 1)  # Keep send queue small
        self._sender.setsockopt(zmq.LINGER, 0)  # Discard unsent messages on close
        self._sender.connect(send_endpoint)

        # SUB socket to receive segmented frames from SAM3
        self._receiver = self._context.socket(zmq.SUB)
        self._receiver.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
        self._receiver.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self._receiver.setsockopt(zmq.RCVTIMEO, 0)  # Non-blocking receive
        self._receiver.connect(recv_endpoint)

    def send_frame(
        self,
        rgb_image: np.ndarray,
        sam3_stage: int,
        prompt: str | None = None
    ) -> bool:
        """
        Send RGB frame to SAM3 server.

        Args:
            rgb_image: RGB image array (H, W, 3), uint8.
            sam3_stage: SAM3 stage counter. Server resets when this increases.
            prompt: Optional text prompt for segmentation.

        Returns:
            True if frame was sent, False otherwise.
        """
        if rgb_image is None:
            return False

        # Store original shape
        self._last_frame_shape = rgb_image.shape

        # Resize to target size expected by SAM3
        target_w, target_h = self.target_size
        if rgb_image.shape[:2] != (target_h, target_w):
            rgb_image = cv2.resize(rgb_image, (target_w, target_h))

        # Ensure uint8 and contiguous for tobytes()
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        rgb_image = np.ascontiguousarray(rgb_image)

        # Use provided prompt or empty string
        actual_prompt = prompt if prompt is not None else ""

        # Pack message with msgpack (single frame, not batched)
        msg = msgpack.packb({
            "prompt": actual_prompt,
            "sam3_stage": sam3_stage,
            "frames": rgb_image.tobytes(),
        }, use_bin_type=True)

        try:
            self._sender.send(msg, zmq.NOBLOCK)

            # Save sent frame for debugging
            if self._output_dir and SAVE_FRAMES:
                sent_path = os.path.join(
                    self._output_dir,
                    "sent_images",
                    f"frame_{self._frame_counter:05d}.png"
                )
                cv2.imwrite(sent_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            return True
        except zmq.Again:
            return False  # Skip if send would block

    def send_model_reset(self):
        """Send model reset signal to SAM3."""
        msg = msgpack.packb({
            "reset_model": True,
        }, use_bin_type=True)
        self._sender.send(msg)

    def drain_stale_messages(self) -> int:
        """Drain any stale messages from the receive buffer.

        This should be called at episode start to clear messages from previous episodes
        that may be buffered due to ZMQ CONFLATE setting.

        Returns:
            Number of messages drained.
        """
        count = 0
        while True:
            try:
                self._receiver.recv(zmq.NOBLOCK)
                count += 1
            except zmq.Again:
                break
        self.latest_segmented_frame = None
        return count

    def receive_segmented_frame(self) -> np.ndarray | None:
        """
        Receive latest segmented frame from SAM3 (non-blocking).

        Returns:
            Segmentation mask as uint8 array (H, W) if available, None otherwise.
            Mask is resized to original_size dimensions.
        """
        try:
            msg = self._receiver.recv(zmq.NOBLOCK)
            target_w, target_h = self.target_size
            expected_pixels = target_h * target_w

            # Infer n_masks from message size
            total_elements = len(msg)
            if total_elements == 0:
                return None

            # Assume single mask for simplicity
            n_masks = total_elements // expected_pixels
            if n_masks == 0:
                return None

            # Reshape to (n_masks, H, W)
            masks = np.frombuffer(msg, dtype=np.uint8).reshape(n_masks, target_h, target_w)

            # Select first mask
            mask = masks[0]  # Shape (H, W)

            # Resize to original size
            original_w, original_h = self.original_size
            if mask.shape != (original_h, original_w):
                mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            # Save received mask for debugging
            if self._output_dir and SAVE_FRAMES:
                # Compute centroid for visualization
                centroid = compute_mask_centroid(mask)
                vis_image = visualize_mask(mask, centroid)

                mask_path = os.path.join(
                    self._output_dir,
                    "received_masks",
                    f"frame_{self._frame_counter:05d}.png"
                )
                cv2.imwrite(mask_path, vis_image)

            self._frame_counter += 1
            self.latest_segmented_frame = mask
            return mask

        except zmq.Again:
            pass  # No new segmented frame available
        except Exception as e:
            print(f"SAM3StreamClient: Error receiving frame: {e}")

        return None

    def close(self):
        """Close ZMQ sockets and terminate context."""
        self._sender.close()
        self._receiver.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Compute the centroid (center of mass) of a binary mask.

    Args:
        mask: Binary mask of shape (H, W), uint8.

    Returns:
        (row, col) centroid coordinates or None if mask is empty.
    """
    if mask is None or mask.sum() == 0:
        return None

    # Find all non-zero points
    coords = np.argwhere(mask > 0)

    # Compute centroid
    centroid_row = int(np.mean(coords[:, 0]))
    centroid_col = int(np.mean(coords[:, 1]))

    return (centroid_row, centroid_col)


def visualize_mask(
    mask: np.ndarray,
    centroid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Create visualization of mask with centroid marked.

    Args:
        mask: Binary segmentation mask (H, W), uint8.
        centroid: (row, col) centroid position.

    Returns:
        Visualization image (BGR format for cv2.imwrite).
    """
    # Convert mask to BGR image (white mask on black background)
    vis_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis_image[mask > 0] = [255, 255, 255]

    # Draw centroid if provided
    if centroid is not None:
        row, col = centroid
        # Draw cross at centroid (red)
        cv2.drawMarker(
            vis_image,
            (col, row),  # cv2 uses (x, y) = (col, row)
            color=(0, 0, 255),  # Red in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )
        # Add text label
        cv2.putText(
            vis_image,
            f"({col}, {row})",
            (col + 10, row - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),  # Red in BGR
            1
        )

    return vis_image


def visualize_mask_with_centroid(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    centroid: Optional[Tuple[int, int]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay mask on RGB image with centroid marked.

    Args:
        rgb_image: Original RGB image (H, W, 3), uint8.
        mask: Binary segmentation mask (H, W), uint8.
        centroid: (row, col) centroid position.
        alpha: Transparency for mask overlay (0=transparent, 1=opaque).

    Returns:
        Visualization image with mask overlay and centroid (RGB format).
    """
    vis_image = rgb_image.copy()

    # Create colored mask overlay (green)
    mask_overlay = np.zeros_like(rgb_image)
    mask_overlay[mask > 0] = [0, 255, 0]  # Green in RGB

    # Blend with original image
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, mask_overlay, alpha, 0)

    # Draw centroid if provided
    if centroid is not None:
        row, col = centroid
        # Draw cross at centroid (red)
        cv2.drawMarker(
            vis_image,
            (col, row),  # cv2 uses (x, y) = (col, row)
            color=(255, 0, 0),  # Red in RGB
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )
        # Add text label
        cv2.putText(
            vis_image,
            f"Centroid: ({col}, {row})",
            (col + 10, row - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),  # Red in RGB
            1
        )

    return vis_image
