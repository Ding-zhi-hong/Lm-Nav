#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from topograph.srv import ExecutePath, ExecutePathResponse

class NavigationController:
    def __init__(self):
        rospy.init_node('nav_controller')
        
        # 参数配置
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.2)  # 米
        self.angular_tolerance = rospy.get_param('~angular_tolerance', 0.2)  # 弧度
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.3)  # 米/秒
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.5)  # 弧度/秒
        self.use_move_base = rospy.get_param('~use_move_base', True)  # 使用move_base或直接控制
        
        # 当前状态
        self.current_pose = None
        self.current_path = None
        self.current_goal_index = 0
        self.active_navigation = False
        
        # ROS接口
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_cb)
        rospy.Subscriber('/nav_path', Path, self.path_cb)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        if self.use_move_base:
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            rospy.loginfo("Waiting for move_base action server...")
            self.move_base_client.wait_for_server()
            rospy.loginfo("Connected to move_base action server")
        
        self.service = rospy.Service('execute_path', ExecutePath, self.execute_path)
        
        rospy.loginfo("Navigation Controller initialized")
    
    def odom_cb(self, msg):
        """里程计回调，更新当前位置"""
        self.current_pose = msg.pose.pose
    
    def amcl_cb(self, msg):
        """AMCL定位回调，提供更准确的位置估计"""
        self.current_pose = msg.pose.pose
    
    def path_cb(self, msg):
        """路径回调，存储接收到的导航路径"""
        if not self.active_navigation:
            self.current_path = msg
            rospy.loginfo(f"Received new navigation path with {len(msg.poses)} waypoints")
    
    def execute_path(self, req):
        """执行接收到的路径"""
        if not self.current_path or len(self.current_path.poses) == 0:
            rospy.logwarn("No valid path to execute")
            return ExecutePathResponse(success=False)
        
        self.active_navigation = True
        self.current_goal_index = 0
        
        # 根据选择的模式执行路径
        if self.use_move_base:
            success = self.execute_with_move_base()
        else:
            success = self.execute_direct_control()
        
        self.active_navigation = False
        return ExecutePathResponse(success=success)
    
    def execute_with_move_base(self):
        """使用move_base执行路径"""
        rospy.loginfo("Starting MoveBase navigation")
        
        # 依次执行每个目标点
        for i, pose_stamped in enumerate(self.current_path.poses):
            self.current_goal_index = i
            goal = MoveBaseGoal()
            goal.target_pose = pose_stamped
            self.move_base_client.send_goal(goal)
            
            # 等待完成或中断
            while not rospy.is_shutdown():
                state = self.move_base_client.get_state()
                if state == 3:  # SUCCEEDED
                    rospy.loginfo(f"Reached waypoint {i+1}/{len(self.current_path.poses)}")
                    break
                elif state >= 4:  # ABORTED 或 REJECTED
                    rospy.logwarn(f"Failed to reach waypoint {i+1}")
                    return False
                
                rospy.sleep(0.1)
        
        rospy.loginfo("Navigation completed")
        return True
    
    def execute_direct_control(self):
        """基于直接控制的导航"""
        rospy.loginfo("Starting direct control navigation")
        
        for i, pose_stamped in enumerate(self.current_path.poses):
            self.current_goal_index = i
            rospy.loginfo(f"Navigating to waypoint {i+1}/{len(self.current_path.poses)}")
            
            # 转向目标方向
            if not self.rotate_to_target(pose_stamped):
                rospy.logwarn(f"Failed to rotate to waypoint {i+1}")
                return False
            
            # 移动到目标位置
            if not self.move_to_target(pose_stamped):
                rospy.logwarn(f"Failed to move to waypoint {i+1}")
                return False
        
        rospy.loginfo("Navigation completed")
        return True
    
    def rotate_to_target(self, target_pose):
        """旋转到目标方向"""
        # 获取目标偏航角
        orientation = target_pose.pose.orientation
        _, _, target_yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        
        # 计算角度差
        while not rospy.is_shutdown() and self.active_navigation:
            if self.current_pose is None:
                rospy.sleep(0.1)
                continue
            
            curr_orientation = self.current_pose.orientation
            _, _, current_yaw = euler_from_quaternion([
                curr_orientation.x,
                curr_orientation.y,
                curr_orientation.z,
                curr_orientation.w
            ])
            
            angle_diff = target_yaw - current_yaw
            
            # 归一化角度差到 [-π, π]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            if abs(angle_diff) < self.angular_tolerance:
                break
            
            # 发送旋转命令
            cmd = Twist()
            cmd.angular.z = np.sign(angle_diff) * min(self.max_angular_speed, abs(angle_diff))
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        # 停止旋转
        self.cmd_vel_pub.publish(Twist())
        return True
    
    def move_to_target(self, target_pose):
        """移动到目标位置"""
        while not rospy.is_shutdown() and self.active_navigation:
            if self.current_pose is None:
                rospy.sleep(0.1)
                continue
            
            # 计算当前位置到目标的距离
            dx = target_pose.pose.position.x - self.current_pose.position.x
            dy = target_pose.pose.position.y - self.current_pose.position.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 计算当前方向到目标方向的角度差
            orientation = self.current_pose.orientation
            _, _, current_yaw = euler_from_quaternion([
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            ])
            
            target_angle = np.arctan2(dy, dx)
            angle_diff = target_angle - current_yaw
            
            # 归一化角度差到 [-π, π]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            if distance < self.position_tolerance:
                break
            
            # 发送移动命令
            cmd = Twist()
            cmd.linear.x = min(self.max_linear_speed, distance * 0.5)  # 比例控制器
            cmd.angular.z = np.sign(angle_diff) * min(self.max_angular_speed, abs(angle_diff))
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        # 停止移动
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(0.5)
        return True

if __name__ == '__main__':
    controller = NavigationController()
    rospy.spin()