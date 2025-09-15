import cv2
import numpy as np
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension

# =======================
# IDS Camera Initialization
# =======================
def initialize_ids_camera():
    try:
        # Initialize the IDS Peak library
        ids_peak.Library.Initialize()

        # Create device manager to access the devices
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        # Check if any devices are connected
        if len(device_manager.Devices()) == 0:
            raise RuntimeError("No IDS camera found.")

        # Open the first camera device
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)

        # Access the node map of the device to configure settings
        nodemap = device.RemoteDevice().NodeMaps()[0]

        # Set the camera's default user set
        nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        nodemap.FindNode("UserSetLoad").Execute()
        nodemap.FindNode("UserSetLoad").WaitUntilDone()

        # Set frame rate (optional)
        try:
            frame_rate_enable_node = nodemap.FindNode("AcquisitionFrameRateEnable")
            if frame_rate_enable_node.IsWritable():
                frame_rate_enable_node.SetValue(True)
                nodemap.FindNode("AcquisitionFrameRate").SetValue(8.0)
        except Exception as e:
            print(f"Warning: Could not set frame rate: {e}")

        # Get camera resolution
        width = int(nodemap.FindNode("Width").Value())
        height = int(nodemap.FindNode("Height").Value())

        # Create data stream for image acquisition
        datastream = device.DataStreams()[0].OpenDataStream()
        payload_size = int(nodemap.FindNode("PayloadSize").Value())
        buffers = []

        # Allocate buffers for image data
        for _ in range(4):
            buffer = datastream.AllocAndAnnounceBuffer(payload_size)
            datastream.QueueBuffer(buffer)
            buffers.append(buffer)

        # Start image acquisition
        datastream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()

        # Initialize image converter for format conversion
        image_converter = ids_peak_ipl.ImageConverter()
        input_pixel_format = ids_peak_ipl.PixelFormat(nodemap.FindNode("PixelFormat").CurrentEntry().Value())
        image_converter.PreAllocateConversion(input_pixel_format, ids_peak_ipl.PixelFormatName_BGRa8, width, height)

        # Return necessary components for the main loop
        return datastream, device, buffers, image_converter, width, height

    except Exception as e:
        print(f"Camera initialization failed: {e}")
        ids_peak.Library.Close()
        return None, None, None, None, None, None

# =======================
# Cleanup and Release Resources
# =======================
def cleanup(datastream, device):
    try:
        if datastream:
            datastream.StopAcquisition()
        if device:
            device.RemoteDevice().NodeMaps()[0].FindNode("AcquisitionStop").Execute()
        ids_peak.Library.Close()
    except Exception as e:
        print(f"Cleanup error: {e}")

# =======================
# Contour Detection with RGB Inside Contours and Inner Thread Contours
# =======================
def process_frame_with_rgb_contours(frame):
    # Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurring (Optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Display the threshold image
    cv2.imshow("Threshold Image", thresh)

    # Finding Contours with hierarchy to detect inner contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering Small Contours
    min_contour_area = 500  # Adjust this value as needed for outer contours
    min_inner_contour_area = 50  # Smaller threshold for inner thread contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Convert the grayscale frame to RGB
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Extract the region inside the outer contours and convert to RGB
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = rgb_frame[y:y + h, x:x + w]
        roi[:] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Ensures the region is in RGB format

    # Drawing outer contours (green) and inner contours (blue)
    contour_img = rgb_frame.copy()
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)  # Green for outer contours

    # Check for threaded/nonthreaded by analyzing contour hierarchy
    is_threaded = False
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Check if contour is outer (no parent)
            if hierarchy[0][i][3] == -1 and cv2.contourArea(cnt) > min_contour_area:
                # Count depth of nested contours
                depth = 1  # Start with outer contour
                current_idx = hierarchy[0][i][2]  # First child
                while current_idx != -1:
                    depth += 1
                    # Move to next child at the same level or deeper
                    next_idx = hierarchy[0][current_idx][2]  # First child of current contour
                    if next_idx == -1:  # No more children, try next sibling
                        next_idx = hierarchy[0][current_idx][0]  # Next contour at same level
                    current_idx = next_idx
                    if depth > 2:  # More than 2 layers (outer + 2 inner)
                        is_threaded = True
                        break
            if is_threaded:
                break

    # Draw inner contours if they exist
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Check if contour has a parent (i.e., is an inner contour) and meets area threshold
            if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > min_inner_contour_area:
                cv2.drawContours(contour_img, [cnt], -1, (255, 0, 0), 1)  # Blue for inner thread contours

    # Add threaded/nonthreaded label to the contour image
    label = "Threaded" if is_threaded else "Nonthreaded"
    cv2.putText(contour_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Region of Interest (ROI) for outer contours
    if len(filtered_contours) > 0:
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]  # Crop the ROI from the frame
            cv2.imshow("ROI", roi)  # Display the ROI

    return contour_img

# =======================
# Run Camera Loop with Contour Detection
# =======================
def run_camera(datastream, device, buffers, image_converter, width, height):
    try:
        while True:
            try:
                # Wait for a buffer to finish (1 second timeout)
                buffer = datastream.WaitForFinishedBuffer(1000)
                if buffer is None:
                    continue
            except ids_peak.Exception as e:
                print(f"Buffer error: {e}")
                continue

            try:
                # Convert the buffer to an image using IDS Peak IPL
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                image_converted = image_converter.Convert(ipl_image, ids_peak_ipl.PixelFormatName_BGRa8)

                # Convert to numpy array for OpenCV processing
                img_array = image_converted.get_numpy_1D()
                frame = np.reshape(img_array, (height, width, 4))  # Ensure the frame is reshaped properly
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Process frame for contours and RGB inside the contours
                contour_frame = process_frame_with_rgb_contours(frame)

                # Display the camera feed with contours
                cv2.imshow("Contours - IDS Camera Feed", contour_frame)

                # Wait for user input ('q' to quit)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] User requested exit.")
                    break

            except Exception as e:
                print(f"Frame processing error: {e}")
            finally:
                # Queue the buffer again for reuse
                datastream.QueueBuffer(buffer)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Cleanup resources on exit
        cv2.destroyAllWindows()
        cleanup(datastream, device)

# =======================
# Main Entry Point
# =======================
def main():
    datastream, device, buffers, image_converter, width, height = initialize_ids_camera()
    if not device:
        print("Failed to initialize camera.")
        return

    run_camera(datastream, device, buffers, image_converter, width, height)

if __name__ == "__main__":
    main()