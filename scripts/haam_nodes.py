#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32, Float32

import os
import sys
import numpy as np

# Add the scripts directory to the Python path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from system_enums import HostState
from rpc_client import DataSubscriber, HeartbeatPublisher, HostStatePublisher

class RosWrapper:
    def __init__(self):
        
        # Get device IP parameter
        self.device_ip = rospy.get_param('~haam_ip', '10.10.10.1')
        
        # Create instances of your existing classes
        self.data_subscriber = DataSubscriber(ip=self.device_ip) 
        self.host_state_manager = HostStatePublisher(ip=self.device_ip)
        self.heartbeat_publisher = HeartbeatPublisher(ip=self.device_ip)
        
        # Create publisher for detection depth
        self.depth_pub = rospy.Publisher('/detection_depth', Float32, queue_size=10)
        
        # Create subscriber for robot state updates
        self.state_sub = rospy.Subscriber('/robot_state_update', Int32, self.state_callback)
        
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
    
    def publish_depth(self):
        """Publish the depth of the first detection if available"""
        with self.data_subscriber.data_lock:
            depth_data = self.data_subscriber.last_depth_data
            if depth_data and depth_data.get('count', 0) > 0 and len(depth_data.get('depths', [])) > 0:
                # DEBUG:
                # Just publish the first detection's depth since max detections set to 1
                depth = Float32(depth_data['depths'][0])
                self.depth_pub.publish(depth)
    
    def start(self):
        """Start all required services"""
        # Start the heartbeat publisher
        self.heartbeat_publisher.start_heartbeat()
        
        # Start data polling
        self.data_subscriber.start_polling()
        
        # Set initial host state
        self.host_state_manager.set_host_state(HostState.IDLE)
        
        # Start publishing loop
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.publish_depth()
            rate.sleep()
    
    def stop(self):
        """Stop all services"""
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
        self.CAMERA_POSITION = np.array([0, 1500])  # [x, y] in mm
        
        # Create subscriber for detection data
        self.depth_detection_sub = rospy.Subscriber("detection_depth", Float32, self.process_detection_depth)
        
        # Create publishers for state commands
        self.start_cmd_pub = rospy.Publisher("pp_start", Int32, queue_size=10)
        self.slow_cmd_pub = rospy.Publisher("pp_slow", Int32, queue_size=10)
        self.stop_cmd_pub = rospy.Publisher("pp_stop", Int32, queue_size=10)
        self.state_pub = rospy.Publisher("/robot_state_update", Int32, queue_size=10)
        
        rospy.loginfo("Detection processor initialized with mode %d", self.mode)
    
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
    
    def process_detection_depth(self, msg):
        """Process incoming depth detection message"""
        depth_mm = msg.data  # Depth is already in mm
        
        # Cam position 1, where HAAM is at top of zone looking down
        if self.mode == 1:
            safety_status = self.check_safety_zone(depth_mm)
            
            if safety_status == 0:
                # SAFE - outside both zones
                rospy.loginfo("Detection at depth %.1f mm: SAFE", depth_mm)
                self.start_cmd_pub.publish(100)  # Full speed
                self.state_pub.publish(HostState.EXECUTE.value)

                
            elif safety_status == 1:
                # SLOW_DOWN - in yellow zone
                rospy.loginfo("Detection at depth %.1f mm: SLOW_DOWN", depth_mm)
                self.slow_cmd_pub.publish(50)   # Reduced speed
                self.state_pub.publish(HostState.HOLDING.value)
                
            elif safety_status == 2:
                # STOP - in red zone
                rospy.loginfo("Detection at depth %.1f mm: STOP", depth_mm)
                self.stop_cmd_pub.publish(0)    # Stop
                self.state_pub.publish(HostState.STOPPED.value)
        
        # Cam pos 2, on arm centered in zones
        elif self.mode == 2:
            pass
            
        # cam pos 3, on base centered in zones
        elif self.mode == 3:
            pass
            
        else:
            rospy.logwarn(f"Unknown mode: {self.mode}")
    
    def start(self):
        """Start the detection processor"""
        rospy.loginfo("Detection processor started")
        rospy.spin()
    
    def stop(self):
        """Stop the detection processor"""
        rospy.loginfo("Detection processor stopped")

if __name__ == "__main__":

    # Initialize ROS node
    rospy.init_node('haam_nodes', anonymous=True)

    wrapper = RosWrapper()
    detection_processor = DetectionProcessor()
    try:
        wrapper.start()
        detection_processor.start()

    except rospy.ROSInterruptException:
        pass

    finally:
        detection_processor.stop()
        wrapper.stop()