#!/usr/bin/env python3
import rospy
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import networkx as nx
import pickle
import torch
import torchvision.transforms as transforms
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse

class TopoNode:
    def __init__(self, node_id, image, position, features):
        """
        拓扑图节点
        :param node_id: 节点ID
        :param image: ROS图像消息或图像路径
        :param position: 位置坐标 (x, y, theta)
        :param features: 特征向量
        """
        self.id = node_id
        self.image = image
        self.position = position
        self.features = features
        self.timestamp = rospy.Time.now()
        self.landmark_probs = {}  # 存储地标概率分布
    
    def get_image_path(self, save_dir):
        """保存节点图像到文件并返回路径"""
        path = os.path.join(save_dir, f"node_{self.id}.jpg")
        cv2.imwrite(path, self.image)
        return path
        
    def set_landmark_probs(self, landmark, probs):
        """设置该节点的地标概率分布"""
        self.landmark_probs[landmark] = probs

class TopoGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.node_counter = 0
        self.feature_dim = 512  # ViNG特征维度
    
    def add_node(self, image, position, features):
        """添加新节点"""
        node_id = self.node_counter
        new_node = TopoNode(node_id, image, position, features)
        self.nodes[node_id] = new_node
        self.graph.add_node(node_id, position=position, features=features)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, node_id1, node_id2, distance):
        """添加边及距离"""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.graph.add_edge(node_id1, node_id2, weight=distance)
    
    def get_neighbors(self, node_id):
        """获取相邻节点"""
        return list(self.graph.neighbors(node_id))
    
    def get_edge_weight(self, node_id1, node_id2):
        """获取边权重"""
        return self.graph[node_id1][node_id2]['weight']
    
    def find_nearest_node(self, position):
        """根据位置寻找最近节点"""
        min_dist = float('inf')
        nearest_id = None
        
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(np.array(node.position[:2]) - np.array(position[:2]))
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        return nearest_id, min_dist
    
    def save(self, filename):
        """保存拓扑图到文件"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'nodes': self.nodes,
                'node_counter': self.node_counter
            }, f)
    
    def load(self, filename):
        """从文件加载拓扑图"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.nodes = data['nodes']
            self.node_counter = data['node_counter']
    


class FeatureExtractor:
    def __init__(self, model_type='clip', device='auto'):
        """
        视觉特征提取器，支持多种预训练模型
        
        参数:
            model_type: 模型类型 ('clip', 'resnet', 'vgg', 'dino')
            device: 计算设备 ('cuda', 'cpu', 'auto')
        """
        rospy.loginfo(f"Initializing FeatureExtractor with model: {model_type}")
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        rospy.loginfo(f"Using device: {self.device}")
        
        # 图像转换器
        self.bridge = CvBridge()
        
        # 加载指定模型
        self.model_type = model_type
        if model_type == 'clip':
            self._load_clip_model()
        else:
            rospy.logerr(f"Unsupported model type: {model_type}. Using CLIP as default.")
            self._load_clip_model()
    
    def _load_clip_model(self):
        """加载CLIP模型"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            rospy.loginfo("CLIP model loaded successfully")
            
            # 设置图像预处理
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
            ])
        except ImportError:
            rospy.logerr("CLIP library not installed. Falling back to ResNet.")
            self._load_resnet_model()
    
    
    def extract(self, image):
        
        # 转换输入为OpenCV图像
        if isinstance(image, str):
            # 图像路径
            cv_image = cv2.imread(image)
            if cv_image is None:
                rospy.logerr(f"Failed to load image from path: {image}")
                return None
        elif isinstance(image, Image):
            # ROS图像消息
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            except Exception as e:
                rospy.logerr(f"Image conversion error: {e}")
                return None
        else:
            # 假设已经是OpenCV格式
            cv_image = image
        
        # 检查图像有效性
        if cv_image is None or cv_image.size == 0:
            rospy.logerr("Invalid image input")
            return None
        
        # 应用预处理
        try:
            if self.model_type == 'clip':
                # CLIP需要特殊处理
                image_tensor = self.transform(cv_image).unsqueeze(0).to(self.device)
            else:
                # 其他模型使用通用预处理
                image_tensor = self.preprocess(cv_image).unsqueeze(0).to(self.device)
        except Exception as e:
            rospy.logerr(f"Image preprocessing failed: {e}")
            return None
        
        # 提取特征
        with torch.no_grad():
            try:
                if self.model_type == 'clip':
                    # CLIP返回图像特征
                    features = self.model.encode_image(image_tensor)
                else:
                    # 其他模型提取特征
                    features = self.model(image_tensor)
                
                # 转换为numpy数组
                features = features.cpu().numpy().flatten()
                return features
            except Exception as e:
                rospy.logerr(f"Feature extraction failed: {e}")
                return None
    def distance(self, features1, features2, distance_type='cosine'):
        """
        计算两个特征向量之间的距离
        
        参数:
            features1: 第一个特征向量
            features2: 第二个特征向量
            distance_type: 距离类型 ('cosine', 'euclidean', 'manhattan')
            
        返回:
            distance: 特征距离 (标量)
        """
        if features1 is None or features2 is None:
            rospy.logwarn("Invalid features for distance calculation")
            return float('inf')
        
        # 确保特征向量形状一致
        if features1.shape != features2.shape:
            rospy.logwarn(f"Feature shape mismatch: {features1.shape} vs {features2.shape}")
            return float('inf')
        
        # 计算指定类型的距离
        if distance_type == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            cosine_sim = dot_product / (norm1 * norm2 + 1e-10)
            return 1.0 - cosine_sim
        
        elif distance_type == 'euclidean':
            # 欧氏距离
            return np.linalg.norm(features1 - features2)
        
        elif distance_type == 'manhattan':
            # 曼哈顿距离
            return np.sum(np.abs(features1 - features2))
        
        else:
            rospy.logwarn(f"Unknown distance type: {distance_type}. Using cosine.")
            return self.distance(features1, features2, 'cosine')    










class VNMTopoMapper:
    def __init__(self):
        rospy.init_node('vnm_topo_mapper')
        
        # 参数配置
        self.map_dir = rospy.get_param('~map_dir', '~/maps')
        self.map_name = rospy.get_param('~map_name', 'office_map')
        self.min_node_distance = rospy.get_param('~min_node_distance', 1.5)  # 米
        self.max_angular = rospy.get_param('~max_angular', 0.5)  # 弧度
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
    
        # 创建输出目录
        self.map_path = os.path.join(os.path.expanduser(self.map_dir), self.map_name)
        os.makedirs(self.map_path, exist_ok=True)
        
        # 初始化组件
        self.bridge = CvBridge()
        self.topo_graph = TopoGraph()
        self.feature_extractor = FeatureExtractor()
        self.last_node_position = None
        self.last_node_features = None
        self.last_node_yaw=None
        self.current_pose = None
        self.current_image=None
        self.current_position=None
        self.current_yaw=None
        # 状态变量
        self.exploring = True
        self.target_yaw = None
        
        # ROS接口
        rospy.Subscriber(self.image_topic, Image, self.image_cb)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        
        # 服务
        rospy.Service('start_exploration', Empty, self.start_exploration)
        rospy.Service('stop_exploration', Empty, self.stop_exploration)
        rospy.Service('save_map', Empty, self.save_map)
        
        rospy.loginfo("VNM TopoMapper initialized. Ready to explore.")
    
    def image_cb(self, msg):
        if not self.exploring:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")
    
    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose
        self.current_position = [
            self.current_pose.position.x,
            self.current_pose.position.y
        ]
        
        # 计算偏航角
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        self.current_yaw = yaw
    
    def start_exploration(self, req):
        self.exploring = True
        rospy.loginfo("Starting environment exploration")
        # 开始探索循环
        rospy.Timer(rospy.Duration(0.1), self.exploration_loop)
        return EmptyResponse()
    
    def stop_exploration(self, req):
        self.exploring = False
        rospy.loginfo("Stopping environment exploration")
        return EmptyResponse()
    
    def save_map(self,req):
        map_file = os.path.join(self.map_path, f"{self.map_name}.graph")
        self.topo_graph.save(map_file)
        rospy.loginfo(f"Topo map saved to {map_file}")
        return EmptyResponse()
    
    def exploration_loop(self, event):
        if not self.exploring or not hasattr(self, 'current_image') or not hasattr(self, 'current_position'):
            return
        
        # 第一次访问，创建起始节点
        if self.last_node_position is None:
            self.create_new_node()
            #self.set_random_direction()
            return
            
        # 计算距离和角度差
        dist = np.linalg.norm(
            np.array(self.current_position) - 
            np.array(self.last_node_position[:2])
        )
        angle_diff = abs(self.current_yaw - self.last_node_yaw)
        
        # 创建新节点的条件
        if dist > self.min_node_distance or angle_diff > self.max_angular:
            self.create_new_node()
        
        # 随机探索控制
        #self.random_exploration()
    
    def create_new_node(self):
        """创建新的拓扑节点"""
        # 提取特征
        features = self.feature_extractor.extract(self.current_image)
        position = [self.current_position[0], self.current_position[1], self.current_yaw]
        
        # 添加到拓扑图
        node_id = self.topo_graph.add_node(
            image=self.current_image,
            position=position,
            features=features
        )
        
        rospy.loginfo(f"Created topology node #{node_id} at ({position[0]:.2f}, {position[1]:.2f})")
        
        # 记录状态
        self.last_node_position = position
        self.last_node_yaw = self.current_yaw
        self.last_node_features = features
        
        # 连接到上一个节点
        if len(self.topo_graph.nodes) > 1:
            prev_node_id = node_id - 1
            distance = self.feature_extractor.distance(
                self.last_node_features,
                features
            )
            self.topo_graph.add_edge(prev_node_id, node_id, distance)
            rospy.loginfo(f"Added edge from {prev_node_id} to {node_id}, distance: {distance:.4f}")
    

if __name__ == '__main__':
    try:
        mapper = VNMTopoMapper()
        rospy.spin()
        # 在关闭时保存地图
        mapper.save_map(EmptyRequest())#
    except rospy.ROSInterruptException:
        pass