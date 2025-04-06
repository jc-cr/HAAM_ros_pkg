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
                        self.depth_pub.publish(depth)
                        rospy.loginfo(f"Published depth: {depth.data} mm")
                    
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
        # Mode based on camera position we are testing. should only be set in the launch
        self.mode = rospy.get_param('~detection_mode', 1)
        
        # Define safety zones in mm
        self.SLOW_DOWN_ZONE_SIZE = 3000  # 3m = 3000mm (yellow square)
        self.STOP_ZONE_SIZE = 2000       # 2m = 2000mm (red square)
        
        # Camera position (at top of yellow square looking down into zones)
        self.CAMERA_POSITION = None 
        
        # Create subscriber for detection data
        self.depth_detection_sub = rospy.Subscriber("/detection_depth", Float32, self.process_detection_depth)
        
        # Create publishers for state commands
        self.start_cmd_pub = rospy.Publisher("pp_start", Int32, queue_size=10)
        self.slow_cmd_pub = rospy.Publisher("pp_slow", Int32, queue_size=10)
        self.stop_cmd_pub = rospy.Publisher("pp_stop", Int32, queue_size=10)
        self.state_pub = rospy.Publisher("/robot_state_update", Int32, queue_size=10)

        self.debug_cmd_executed_pub = rospy.Publisher("cmd_executed_timestamp", Int64, queue_size=10)
        
        # Add timestamp publisher - ensure it's Int64
        self.cmd_sent_pub = rospy.Publisher("/cmd_sent_timestamp", Int64, queue_size=10)
        
        rospy.loginfo("Detection processor initialized with mode %d", self.mode)

    def process_detection_depth(self, msg):
        """Process incoming depth detection message"""
        depth_mm = msg.data  # Depth is already in mm
        
        # Add debug output
        rospy.logdebug(f"Processing detection at depth: {depth_mm} mm")
        
        # Initialize camera position based on mode
        if self.mode == 1:
            self.CAMERA_POSITION = np.array([0, 1500])  # [x, y] in mm
        
        # Cam position 1, where HAAM is at top of zone looking down
        if self.mode == 1:
            safety_status = self.check_safety_zone(depth_mm)
            
            # Add more verbose output
            rospy.logdebug(f"Safety status: {safety_status} (0=SAFE, 1=SLOW, 2=STOP)")
            
            if safety_status == 0:
                # SAFE - outside both zones
                rospy.logdebug("Detection at depth %.1f mm: SAFE", depth_mm)
                self.start_cmd_pub.publish(100)  # Full speed
                self.state_pub.publish(HostState.EXECUTE.value)
                
            elif safety_status == 1:
                # SLOW_DOWN - in yellow zone
                rospy.logdebug("Detection at depth %.1f mm: SLOW_DOWN", depth_mm)
                self.slow_cmd_pub.publish(50)   # Reduced speed
                self.state_pub.publish(HostState.HOLDING.value)
                
            elif safety_status == 2:
                # STOP - in red zone
                rospy.logdebug("Detection at depth %.1f mm: STOP", depth_mm)
                self.stop_cmd_pub.publish(0)    # Stop

                self.state_pub.publish(HostState.STOPPED.value)


                # Create timestamp for command sent
                timestamp = rospy.Time.now().to_nsec()
                self.cmd_sent_pub.publish(timestamp)
                rospy.loginfo(f"Published command sent timestamp: {timestamp}")

        
        # Debug mode added from the full haam_nodes.py
        elif self.mode == 4:
            rospy.logdebug(f"Debug mode - depth: {depth_mm} mm")
            if depth_mm > 3000:
                # SAFE - outside both zones
                rospy.loginfo("Detection at depth %.1f mm: SAFE", depth_mm)
                self.start_cmd_pub.publish(100)  # Full speed
                self.state_pub.publish(HostState.EXECUTE.value)
                
            elif depth_mm <= 3000 and depth_mm >= 1000:  # Fixed condition
                # SLOW_DOWN - in yellow zone
                rospy.loginfo("Detection at depth %.1f mm: SLOW_DOWN", depth_mm)
                self.slow_cmd_pub.publish(50)   # Reduced speed
                self.state_pub.publish(HostState.HOLDING.value)
                
            elif depth_mm < 1000:
                # STOP - in red zone
                rospy.loginfo("Detection at depth %.1f mm: STOP", depth_mm)
                self.stop_cmd_pub.publish(0)    # Stop
                self.state_pub.publish(HostState.STOPPED.value)

                # Create timestamp for command sent
                timestamp = rospy.Time.now().to_nsec()
                self.cmd_sent_pub.publish(timestamp)
                rospy.loginfo(f"Published command sent timestamp: {timestamp}")

                # Simulate timestamp for robto execution

                timestamp = rospy.Time.now().to_nsec()
                self.debug_cmd_executed_pub.publish(timestamp)
                rospy.loginfo(f"Published command sent timestamp: {timestamp}")


            
        
        else:
            rospy.logwarn(f"Unknown mode: {self.mode}")
            


    def check_point_in_square(self, point, half_size):
        """Check if a point is inside a square centered at origin"""
        x, y = point
        return -half_size <= x <= half_size and -half_size <= y <= half_size
    
    def project_point_from_depth(self, depth_mm):
        """
        For mode 1, project the depth reading onto the workspace plane.
        
        Since the camera is at (0, 1500) looking into the workspace,
        a depth reading represents a point along the negative y-axis
        from the camera position.
        """
        return np.array([0, self.CAMERA_POSITION[1] - depth_mm])
    
    def check_safety_zone(self, depth_mm):
        """
        Check which safety zone a detection is in based on its projected position.
        
        Args:
            depth_mm: Depth measurement in mm from the camera
            
        Returns:
            0: SAFE (outside both zones)
            1: SLOW_DOWN (in yellow zone but not in red zone)
            2: STOP (in red zone)
        """
        # Project the depth onto the workspace
        point = self.project_point_from_depth(depth_mm)
        
        # Check if point is in STOP zone (red square)
        half_stop = self.STOP_ZONE_SIZE / 2
        if self.check_point_in_square(point, half_stop):
            return 2
        
        # Check if point is in SLOW_DOWN zone (yellow square)
        half_slow = self.SLOW_DOWN_ZONE_SIZE / 2
        if self.check_point_in_square(point, half_slow):
            return 1
        
        # Outside both zones
        return 0


if __name__ == "__main__":
    # Set log level to DEBUG for more verbose output
    rospy.init_node('haam_nodes', anonymous=True, log_level=rospy.INFO)
    
    rospy.loginfo("Starting HAAM nodes")
    
    try:
        # Initialize components with more verbose output
        rospy.loginfo("Initializing RosWrapper...")
        wrapper = RosWrapper()
        
        rospy.loginfo("Initializing DetectionProcessor...")
        detection_processor = DetectionProcessor()
        
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
    # ...rest of the exception handling