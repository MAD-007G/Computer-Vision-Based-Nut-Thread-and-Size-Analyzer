import cv2
import numpy as np
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension
import time

# ========== IDS Camera Setup ==========

def initialize_ids_camera():
    try:
        # Initialize IDS Peak library
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        # Check if any devices are available
        if not device_manager.Devices():
            raise Exception("No IDS cameras found")

        # Open the first available device
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        nodemap = device.RemoteDevice().NodeMaps()[0]

        # Load default settings
        nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        nodemap.FindNode("UserSetLoad").Execute()
        nodemap.FindNode("UserSetLoad").WaitUntilDone()

        # Set frame rate to 24 FPS if supported
        try:
            nodemap.FindNode("AcquisitionFrameRateEnable").SetValue(True)
            nodemap.FindNode("AcquisitionFrameRate").SetValue(24.0)
        except Exception as e:
            print(f"Warning: Could not set frame rate: {e}")

        # Set resolution to maximum or 3840x2160
        max_width = nodemap.FindNode("WidthMax").Value()
        max_height = nodemap.FindNode("HeightMax").Value()
        nodemap.FindNode("Width").SetValue(min(1280, max_width))
        nodemap.FindNode("Height").SetValue(min(720, max_height))

        # Initialize data stream
        datastream = device.DataStreams()[0].OpenDataStream()
        payload_size = nodemap.FindNode("PayloadSize").Value()
        buffers = [datastream.AllocAndAnnounceBuffer(payload_size) for _ in range(datastream.NumBuffersAnnouncedMinRequired())]
        for buf in buffers:
            datastream.QueueBuffer(buf)

        # Start acquisition
        datastream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()

        # Initialize image converter
        image_converter = ids_peak_ipl.ImageConverter()
        input_fmt = ids_peak_ipl.PixelFormat(nodemap.FindNode("PixelFormat").CurrentEntry().Value())
        image_converter.PreAllocateConversion(input_fmt, ids_peak_ipl.PixelFormatName_BGRa8, max_width, max_height)

        return datastream, device, buffers, image_converter

    except Exception as e:
        print(f"Error initializing camera: {e}")
        ids_peak.Library.Close()
        raise

# ========== Frame Processing ==========

def process_frame(frame, min_contour_area=500, debug=False, known_object_distance_cm=49.0, grid_spacing_cm=1.0):
    # Camera parameters (replace with actual values from your camera/lens)
    FOCAL_LENGTH_MM = 8.0  # Example focal length in mm (check lens specs)
    SENSOR_PIXEL_SIZE_MM = 0.0022  # Example pixel size in mm/pixel (check camera specs)
    OBJECT_DISTANCE_MM = known_object_distance_cm * 10.0  # Convert cm to mm

    # Conversion factor: cm per pixel
    conversion_factor_cm = (OBJECT_DISTANCE_MM * SENSOR_PIXEL_SIZE_MM) / (FOCAL_LENGTH_MM * 10.0)  # cm/pixel
    # Apply calibration factor to correct measurement (4.35 cm -> 1.45 cm, so divide by ~3)
    calibration_factor = 1.0 / 3.0  # Adjust based on observed discrepancy
    conversion_factor_cm *= calibration_factor

    # Pixels per cm for grid
    pixels_per_cm = 1.0 / conversion_factor_cm if conversion_factor_cm > 0 else 1.0

    # Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blurring
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Finding Contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering Contours (nut-like shapes: hexagonal or circular)
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        # Filter for hexagonal or circular shapes
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        if len(approx) >= 6 or (0.7 < circularity < 1.0):  # Hexagon or near-circular
            filtered_contours.append(cnt)

    # Draw Contours
    output_frame = frame.copy()
    cv2.drawContours(output_frame, filtered_contours, -1, (0, 255, 0), thickness=2)

    # Centered Detection Box
    frame_height, frame_width = frame.shape[:2]
    box_width, box_height = 600, 600
    detection_box_top_left = ((frame_width - box_width) // 2, (frame_height - box_height) // 2)
    detection_box_bottom_right = (detection_box_top_left[0] + box_width, detection_box_top_left[1] + box_height)
    cv2.rectangle(output_frame, detection_box_top_left, detection_box_bottom_right, (255, 0, 0), 2)

    # Process Objects in the Box
    detected_nut = None
    max_area = 0
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Check if contour is inside the detection box
        if (detection_box_top_left[0] <= x <= detection_box_bottom_right[0] and
                detection_box_top_left[1] <= y <= detection_box_bottom_right[1]):
            area = cv2.contourArea(cnt)
            if area > max_area:  # Select the largest contour in the box
                max_area = area
                detected_nut = cnt

    # Calculate and Display Dimensions with Laser Bars and Grid
    if detected_nut is not None:
        x, y, w, h = cv2.boundingRect(detected_nut)
        # Convert pixel dimensions to physical dimensions (cm)
        width_cm = w * conversion_factor_cm
        height_cm = h * conversion_factor_cm
        print(f"Detected Nut Width: {width_cm:.2f} cm")
        print(f"Detected Nut Height: {height_cm:.2f} cm")

        # Center of the object
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw horizontal red laser bar for width
        cv2.line(output_frame, (x, center_y), (x + w, center_y), (0, 0, 255), 2)
        # Draw vertical red laser bar for height
        cv2.line(output_frame, (center_x, y), (center_x, y + h), (0, 0, 255), 2)

        if debug:
            # Display width text near the horizontal line
            cv2.putText(output_frame, f"Width: {width_cm:.2f} cm", (x + w + 10, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Display height text near the vertical line
            cv2.putText(output_frame, f"Height: {height_cm:.2f} cm", (center_x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == "__main__":

    main()
