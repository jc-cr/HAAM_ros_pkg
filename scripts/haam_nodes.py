#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32, Int64, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import os
import sys
import json
import numpy as np
import cv2
import threading
from datetime import datetime
from PIL import Image as PILImage

# Add the scripts directory to the Python path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from system_enums import HostState
from rpc_client import DataSubscriber, HeartbeatPublisher, HostStatePublisher

class DataLogging:
    def __init__(self):

        # Ensure logs directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)  # Parent directory of scripts
        self.base_log_dir = os.path.join(package_dir, "logs")
        os.makedirs(self.base_log_dir, exist_ok=True)
        rospy.loginfo(f"Logging data to: {self.base_log_dir}")
        
        # Bridge for converting ROS images to OpenCV/PIL
        self.bridge = CvBridge()
        
        # Subscribe to command sent timestamps
        self.cmd_sent_sub = rospy.Subscriber('/cmd_sent_timestamp', Int64, self.cmd_sent_callback)
        
        # Subscribe to robot execution timestamps
        self.cmd_executed_sub = rospy.Subscriber('/cmd_executed_timestamp', Int64, self.cmd_executed_callback)
        
        # Subscribe to detection frame
        self.detection_frame_sub = rospy.Subscriber('/detection_frame', Image, self.detection_frame_callback)
        
        # Subscribe to depth for additional context
        self.depth_sub = rospy.Subscriber('/detection_depth', Float32, self.depth_callback)
        
        # Initialize variables to track most recent data
        self.last_cmd_sent_time = None
        self.last_cmd_executed_time = None
        self.last_frame = None
        self.last_depth_mm = None
        
        rospy.loginfo("Data logging initialized")
    
    def cmd_sent_callback(self, msg):
        """Callback for command sent timestamps"""
        # Just store the timestamp as is
        self.last_cmd_sent_time = msg.data
        rospy.loginfo(f"Received command sent timestamp: {self.last_cmd_sent_time}")
    
    def cmd_executed_callback(self, msg):
        """Callback for command executed timestamps"""
        # Store the execution timestamp
        self.last_cmd_executed_time = msg.data
        rospy.loginfo(f"Received command executed timestamp: {self.last_cmd_executed_time}")
        
        # Log STOP data if we have all the necessary information
        self.log_stop_data()
    
    def detection_frame_callback(self, msg):
        """Callback for detection frame"""
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Convert to PIL Image
            self.last_frame = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            rospy.logdebug("Received detection frame")
            
        except Exception as e:
            rospy.logerr(f"Error processing detection frame: {e}")
    
    def depth_callback(self, msg):
        """Callback for depth data"""
        self.last_depth_mm = msg.data
    
    def log_stop_data(self):
        """Log data for a STOP event"""
        # Check if we have all necessary data
        if (self.last_cmd_sent_time is None or 
            self.last_cmd_executed_time is None or 
            self.last_frame is None):
            rospy.logwarn("Cannot log STOP data: missing timestamp or frame")
            return
        
        try:
            # Create timestamped directory for this STOP event
            stop_dir = os.path.join(self.base_log_dir, f"stop_{self.last_cmd_executed_time}")
            os.makedirs(stop_dir, exist_ok=True)
            
            # Calculate stopping time in milliseconds
            stopping_time_ns = self.last_cmd_executed_time - self.last_cmd_sent_time
            stopping_time_ms = stopping_time_ns / 1000000  # Convert to milliseconds
            
            # Create JSON with timestamp and stopping time
            json_data = {
                "system_timestamp": self.last_cmd_executed_time,
                "stopping_time_ms": stopping_time_ms,
                "depth_mm": self.last_depth_mm if self.last_depth_mm is not None else 0
            }
            
            # Write JSON file
            json_path = os.path.join(stop_dir, "stop_data.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save inference image
            img_path = os.path.join(stop_dir, "inference_image.jpg")
            self.last_frame.save(img_path)
            
            rospy.loginfo(f"Logged STOP data to {stop_dir} - Stopping time: {stopping_time_ms} ms")
            
            # Reset data after logging
            self.last_cmd_sent_time = None
            self.last_cmd_executed_time = None
            self.last_frame = None
            # Don't reset depth, it's useful to keep the latest
            
        except Exception as e:
            rospy.logerr(f"Error logging STOP data: {e}")


class RosWrapper:
    def __init__(self):
        # Get device IP parameter
        self.device_ip = rospy.get_param('~haam_ip', '10.10.10.1')
        
        # Create bridge for converting PIL to ROS images
        self.bridge = CvBridge()
        
        # Create instances of your existing classes
        self.data_subscriber = DataSubscriber(ip=self.device_ip) 
        self.host_state_manager = HostStatePublisher(ip=self.device_ip)
        self.heartbeat_publisher = HeartbeatPublisher(ip=self.device_ip)

        # Create publisher for detection depth and frame
        self.depth_pub = rospy.Publisher('/detection_depth', Float32, queue_size=10)
        self.frame_pub = rospy.Publisher('/detection_frame', Image, queue_size=10)
        
        # Create subscriber for robot state updates
        self.state_sub = rospy.Subscriber('/robot_state_update', Int32, self.state_callback)
        
        # For clean shutdown
        self.running = False
        self.main_thread = None
        
        rospy.loginfo("HAAM ROS wrapper nodes initialized")
    
    def state_callback(self, msg):
        """Handle robot state update from ROS"""
        state_value = msg.data
        try:
            state = HostState(state_value)
            success = self.host_state_manager.set_host_state(state)
            if success:
                rospy.loginfo(f"Set host state to {state.name}")
            else:
                rospy.logwarn(f"Failed to set host state to {state.name}")
        except ValueError:
            rospy.logerr(f"Invalid state value: {state_value}")


    def _worker_thread(self):
        """Main worker thread that polls data and publishes it"""
        rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("Worker thread started - polling at 10Hz")
        
        poll_count = 0
        while self.running and not rospy.is_shutdown():
            try:
                # Log heartbeat periodically
                poll_count += 1
                if poll_count % 20 == 0:  # Log every ~2 seconds
                    rospy.logdebug(f"Worker thread alive, poll count: {poll_count}")
                
                # Get latest data with better error handling
                try:
                    latest_data = self.data_subscriber.get_latest_data()
                except Exception as e:
                    rospy.logerr(f"Error getting latest data: {e}")
                    latest_data = None
                
                if latest_data:
                    # Process depth data
                    if (latest_data['depth_data'] and 
                        latest_data['depth_data'].get('count', 0) > 0 and 
                        len(latest_data['depth_data'].get('depths', [])) > 0):
                        
                        depth = Float32(latest_data['depth_data']['depths'][0])
                        if depth.data > 0:
                            self.depth_pub.publish(depth)
                            rospy.logdebug(f"Published depth: {depth.data} mm")

                        else:
                            continue
                    
                    # Process frame data
                    if latest_data['frame'] is not None:
                        try:
                            cv_image = np.array(latest_data['frame'])
                            # Handle RGB/BGR conversion safely
                            cv_image = cv_image[:, :, ::-1].copy() if cv_image.ndim == 3 else cv_image
                            
                            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                            ros_image.header.stamp = rospy.Time.now()
                            
                            self.frame_pub.publish(ros_image)
                            rospy.logdebug("Published frame")
                        except Exception as e:
                            rospy.logerr(f"Error publishing frame: {e}")
                    
                    # Log system state and detection info
                    detection_count = latest_data['detection_data'].get('count', 0) if latest_data['detection_data'] else 0
                    system_state_name = latest_data['system_state'].name if hasattr(latest_data['system_state'], 'name') else str(latest_data['system_state'])
                    
                    rospy.logdebug(f"System state: {system_state_name}, Detection count: {detection_count}")
                else:
                    rospy.logwarn("No data available from data subscriber")
            
            except Exception as e:
                rospy.logerr(f"Error in worker thread: {e}")
            
            # Sleep at the specified rate
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                self.running = False
                break
        
    def start(self):
        """Start all required services"""
        # Start the heartbeat publisher
        self.heartbeat_publisher.start_heartbeat()
        
        # Start data polling
        self.data_subscriber.start_polling()
        
        # Set initial host state
        self.host_state_manager.set_host_state(HostState.IDLE)
        
        # Start the worker thread
        self.running = True
        self.main_thread = threading.Thread(target=self._worker_thread)
        self.main_thread.daemon = True
        self.main_thread.start()
        rospy.loginfo("Started worker thread")
    
    def stop(self):
        """Stop all services"""
        self.running = False
        
        # Wait for the thread to finish
        if self.main_thread and self.main_thread.is_alive():
            rospy.loginfo("Waiting for worker thread to stop...")
            self.main_thread.join(timeout=2.0)
        
        # Stop background services
        self.heartbeat_publisher.stop_heartbeat()
        self.data_subscriber.stop_polling()
        rospy.loginfo("HAAM ROS services stopped")

class DetectionProcessor:
    def __init__(self):
        # Get all parameters from ROS parameter server with defaults

        # Mode based on camera position we are testing
        self.mode = rospy.get_param('~detection_mode', 4)
        
        # Define safety zones in mm (base boundaries)
        self.RED_BOUNDARY_X = rospy.get_param('~red_boundary_mm', 1000)
        self.YELLOW_BOUNDARY_X = rospy.get_param('~yellow_boundary_mm', 1500)
        
        # Define camera heights for different views (in mm)
        self.AERIAL_HEIGHT = rospy.get_param('~aerial_height_mm', 2000)
        self.NORMAL_HEIGHT = rospy.get_param('~normal_height_mm', 1580)
        self.GROUND_HEIGHT = rospy.get_param('~ground_height_mm', 400)
        
        # Current camera height (will be set based on mode)
        self.camera_height = None
        
        # Hysteresis margin (in mm) - this creates a "dead zone" to prevent oscillation
        self.HYSTERESIS_MARGIN = rospy.get_param('~hysteresis_margin_mm', 100)
        
        # Track current state to implement hysteresis
        # 0 = SAFE, 1 = SLOW, 2 = STOP
        self.current_state = 0  # Initialize to SAFE state
        
        # Initialize camera position based on mode
        self._setup_camera_mode()
        
        # Calculate depth thresholds using the Euclidean distance formula
        # depth = sqrt(x_position^2 + camera_height^2)
        if self.mode in [1, 2, 3]:
            # Calculate regular thresholds
            self.red_threshold = np.sqrt(self.RED_BOUNDARY_X**2 + self.camera_height**2)
            self.yellow_threshold = np.sqrt(self.YELLOW_BOUNDARY_X**2 + self.camera_height**2)
            
            # Calculate exit thresholds with hysteresis margin
            self.red_exit_threshold = self.red_threshold + self.HYSTERESIS_MARGIN
            self.yellow_exit_threshold = self.yellow_threshold + self.HYSTERESIS_MARGIN
        elif self.mode == 4:  # Debug mode
            self.red_threshold = 1000
            self.yellow_threshold = 3000
            self.red_exit_threshold = self.red_threshold + self.HYSTERESIS_MARGIN
            self.yellow_exit_threshold = self.yellow_threshold + self.HYSTERESIS_MARGIN
        
        # Create subscriber for detection data
        self.depth_detection_sub = rospy.Subscriber("/detection_depth", Float32, self.process_detection_depth)
        
        # Create publishers for state commands
        self.start_cmd_pub = rospy.Publisher("pp_start", Int32, queue_size=10)
        self.slow_cmd_pub = rospy.Publisher("pp_slow", Int32, queue_size=10)
        self.stop_cmd_pub = rospy.Publisher("pp_stop", Int32, queue_size=10)
        self.state_pub = rospy.Publisher("robot_state_update", Int32, queue_size=10)

        self.debug_cmd_executed_pub = rospy.Publisher("cmd_executed_timestamp", Int64, queue_size=10)
        
        # Add timestamp publisher - ensure it's Int64
        self.cmd_sent_pub = rospy.Publisher("cmd_sent_timestamp", Int64, queue_size=10)
        
        # Log the calculated thresholds for verification
        self._log_thresholds()
    
    def _setup_camera_mode(self):
        """Setup camera parameters based on selected mode"""
        if self.mode == 1:  # Aerial view
            self.camera_height = self.AERIAL_HEIGHT
            rospy.loginfo("Mode 1: Aerial view camera (height: %d mm)", self.camera_height)
        elif self.mode == 2:  # Normal view
            self.camera_height = self.NORMAL_HEIGHT
            rospy.loginfo("Mode 2: Normal view camera (height: %d mm)", self.camera_height)
        elif self.mode == 3:  # Ground view
            self.camera_height = self.GROUND_HEIGHT
            rospy.loginfo("Mode 3: Ground view camera (height: %d mm)", self.camera_height)
        elif self.mode == 4:  # Debug mode (using direct thresholds)
            rospy.loginfo("Mode 4: Debug mode (using hardcoded thresholds)")
            return
        else:
            rospy.logwarn("Unknown mode: %d. Defaulting to mode 1 (Aerial view)", self.mode)
            self.mode = 1
            self.camera_height = self.AERIAL_HEIGHT
    
    def _log_thresholds(self):
        """Log the calculated depth thresholds for verification"""
        if self.mode in [1, 2, 3]:
            rospy.loginfo("===== CALCULATED DEPTH THRESHOLDS WITH HYSTERESIS =====")
            rospy.loginfo("Camera height: %d mm", self.camera_height)
            rospy.loginfo("RED zone boundary (x): %d mm", self.RED_BOUNDARY_X)
            rospy.loginfo("YELLOW zone boundary (x): %d mm", self.YELLOW_BOUNDARY_X)
            rospy.loginfo("Hysteresis margin: %d mm", self.HYSTERESIS_MARGIN)
            rospy.loginfo("Calculation formula: depth = sqrt(x_boundary^2 + camera_height^2)")
            rospy.loginfo("RED zone entry threshold: %.1f mm", self.red_threshold)
            rospy.loginfo("RED zone exit threshold: %.1f mm", self.red_exit_threshold)
            rospy.loginfo("YELLOW zone entry threshold: %.1f mm", self.yellow_threshold)
            rospy.loginfo("YELLOW zone exit threshold: %.1f mm", self.yellow_exit_threshold)
            rospy.loginfo("===================================================")
        elif self.mode == 4:
            rospy.loginfo("===== DEBUG MODE THRESHOLDS WITH HYSTERESIS =====")
            rospy.loginfo("Hysteresis margin: %d mm", self.HYSTERESIS_MARGIN)
            rospy.loginfo("RED zone entry threshold: %.1f mm", self.red_threshold)
            rospy.loginfo("RED zone exit threshold: %.1f mm", self.red_exit_threshold)
            rospy.loginfo("YELLOW zone entry threshold: %.1f mm", self.yellow_threshold)
            rospy.loginfo("YELLOW zone exit threshold: %.1f mm", self.yellow_exit_threshold)
            rospy.loginfo("===================================================")
    
    def process_detection_depth(self, msg):
        """Process incoming depth detection message with hysteresis"""
        depth_mm = msg.data  # Depth is already in mm
        
        # Determine new state with hysteresis
        new_state = self.determine_state_with_hysteresis(depth_mm)
        
        # Only take action if the state has changed
        if new_state != self.current_state:
            rospy.loginfo("State change: %d -> %d at depth %.1f mm", 
                         self.current_state, new_state, depth_mm)
            
            # Update current state
            self.current_state = new_state
            
            # Take action based on new state
            if new_state == 0:  # SAFE
                rospy.loginfo("Detection at depth %.1f mm: SAFE", depth_mm)
                self.start_cmd_pub.publish(100)  # Full speed
                self.state_pub.publish(HostState.EXECUTE.value)
                
            elif new_state == 1:  # SLOW_DOWN
                rospy.loginfo("Detection at depth %.1f mm: SLOW_DOWN", depth_mm)
                self.slow_cmd_pub.publish(50)  # Reduced speed
                self.state_pub.publish(HostState.HOLDING.value)
                
            elif new_state == 2:  # STOP
                rospy.loginfo("Detection at depth %.1f mm: STOP", depth_mm)
                self.stop_cmd_pub.publish(0)  # Stop
                self.state_pub.publish(HostState.STOPPED.value)
                
                # Create timestamp for command sent
                timestamp = rospy.Time.now().to_nsec()
                self.cmd_sent_pub.publish(timestamp)
                rospy.loginfo(f"Published command sent timestamp: {timestamp}")
                
                # In debug mode, also publish execution timestamp
                if self.mode == 4:
                    timestamp = rospy.Time.now().to_nsec()
                    self.debug_cmd_executed_pub.publish(timestamp)
                    rospy.loginfo(f"Published command executed timestamp: {timestamp}")
    
    def determine_state_with_hysteresis(self, depth_mm):
        """Determine safety state with hysteresis to prevent oscillation
        
        This function implements the following hysteresis logic:
        1. If currently in SAFE state (0):
           - Only transition to SLOW (1) if depth <= yellow_threshold
           - Only transition to STOP (2) if depth <= red_threshold
        
        2. If currently in SLOW state (1):
           - Transition to SAFE (0) only if depth > yellow_exit_threshold
           - Transition to STOP (2) if depth <= red_threshold
        
        3. If currently in STOP state (2):
           - Transition to SLOW (1) only if depth > red_exit_threshold
           - Transition to SAFE (0) only if depth > yellow_exit_threshold
        
        Returns: 
            0 - SAFE (outside both zones)
            1 - SLOW_DOWN (in yellow zone) 
            2 - STOP (in red zone)
        """
        # Current state is SAFE (0)
        if self.current_state == 0:
            if depth_mm <= self.red_threshold:
                return 2  # Go to STOP
            elif depth_mm <= self.yellow_threshold:
                return 1  # Go to SLOW
            else:
                return 0  # Stay in SAFE
        
        # Current state is SLOW (1)
        elif self.current_state == 1:
            if depth_mm <= self.red_threshold:
                return 2  # Go to STOP
            elif depth_mm > self.yellow_exit_threshold:
                return 0  # Go to SAFE (requires crossing exit threshold)
            else:
                return 1  # Stay in SLOW
        
        # Current state is STOP (2)
        elif self.current_state == 2:
            if depth_mm > self.yellow_exit_threshold:
                return 0  # Go directly to SAFE if well outside yellow zone
            elif depth_mm > self.red_exit_threshold:
                return 1  # Go to SLOW if outside red zone + hysteresis
            else:
                return 2  # Stay in STOP
        
        # Fallback (should never reach here)
        rospy.logwarn("Invalid current_state: %d. Defaulting to SAFE.", self.current_state)
        return 0




if __name__ == "__main__":
    # Set log level to DEBUG for more verbose output
    rospy.init_node('haam_nodes', anonymous=True, log_level=rospy.INFO)
    
    rospy.loginfo("Starting HAAM nodes")

    log_data_flag = rospy.get_param('~log_data', 0)
    
    try:
        # Initialize components with more verbose output
        rospy.loginfo("Initializing RosWrapper...")
        wrapper = RosWrapper()
        
        rospy.loginfo("Initializing DetectionProcessor...")
        detection_processor = DetectionProcessor()
        
        if log_data_flag:
            rospy.loginfo("Initializing DataLogging...")
            data_logger = DataLogging()

        # Start components
        rospy.loginfo("Starting RosWrapper services...")
        wrapper.start()
        
        # Let ROS handle the callbacks
        rospy.loginfo("HAAM nodes running. Press Ctrl+C to terminate.")
        rospy.spin()
        
    except Exception as e:
        rospy.logerr(f"Error during HAAM nodes execution: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())