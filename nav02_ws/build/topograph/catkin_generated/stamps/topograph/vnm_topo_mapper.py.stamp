#!/usr/bin/env python3
import numpy as np
import networkx as nx
import os
import cv2
import rospy
import dill
import copy
import math
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image
from sklearn.neighbors import KDTree
import torch
import torchvision.transforms as transforms
import clip

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
        self.image = image  # 存储为 numpy 数组
        self.position = position  # [x, y, theta] 列表
        self.features = features  # numpy 数组
        self.timestamp = rospy.Time.now().to_sec()  # 存储为浮点数
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
        self.graph = nx.Graph()
        self.nodes = {}
        self.node_counter = 0
        self.feature_dim = 512  # ViNG特征维度
    
    def add_node(self, image, position, features):
        """添加新节点"""
        node_id = self.node_counter
        features = features.flatten()
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
        # 创建可序列化的数据字典
        serializable_data = {
            'graph': self._serialize_graph(),
            'nodes': self._serialize_nodes(),
            'node_counter': self.node_counter
        }
        
        with open(filename, 'wb') as f:
            dill.dump(serializable_data, f)
    
    def _serialize_graph(self):
        """将图结构转换为可序列化的字典"""
        serialized_graph = {
            'nodes': {},
            'edges': []
        }
        
        # 序列化节点属性
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            serialized_graph['nodes'][node_id] = {
                'position': node_data['position'],
                'features': node_data['features'].tolist()  # 转换为列表
            }
        
        # 序列化边
        for u, v, data in self.graph.edges(data=True):
            serialized_graph['edges'].append({
                'source': u,
                'target': v,
                'weight': data['weight']
            })
        
        return serialized_graph
    
    def _serialize_nodes(self):
        """将节点字典转换为可序列化的字典"""
        serialized_nodes = {}
        for node_id, node in self.nodes.items():
            serialized_nodes[node_id] = {
                'id': node.id,
                'image': node.image.tolist() if isinstance(node.image, np.ndarray) else node.image,
                'position': node.position,
                'features': node.features.tolist(),
                'timestamp': node.timestamp,
                'landmark_probs': node.landmark_probs
            }
        return serialized_nodes
    
    def load(self, filename):
        """从文件加载拓扑图"""
        with open(filename, 'rb') as f:
            data = dill.load(f)
            self.node_counter = data['node_counter']
            self.graph = self._deserialize_graph(data['graph'])
            self.nodes = self._deserialize_nodes(data['nodes'])
    
    def _deserialize_graph(self, serialized_graph):
        """从序列化数据重建图结构"""
        graph = nx.Graph()
        
        # 重建节点
        for node_id, node_data in serialized_graph['nodes'].items():
            graph.add_node(node_id, 
                           position=node_data['position'],
                           features=np.array(node_data['features']))
        
        # 重建边
        for edge in serialized_graph['edges']:
            graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        return graph
    
    def _deserialize_nodes(self, serialized_nodes):
        """从序列化数据重建节点字典"""
        nodes = {}
        for node_id, node_data in serialized_nodes.items():
            node = TopoNode(
                node_id=node_data['id'],
                image=np.array(node_data['image']) if isinstance(node_data['image'], list) else node_data['image'],
                position=node_data['position'],
                features=np.array(node_data['features'])
            )
            node.timestamp = node_data['timestamp']
            node.landmark_probs = node_data['landmark_probs']
            nodes[node_id] = node
        return nodes





import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import clip  # 确保已安装openai-clip包

class FeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device='auto', normalize=True):
        """
        CLIP特征提取器
        
        参数:
            model_name: CLIP模型名称 (默认'ViT-B/32')
            device: 计算设备 ('cuda', 'cpu', 'auto')
            normalize: 是否对特征向量进行L2归一化
        """
        # 设置设备
        self.device = device
        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # 加载CLIP模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"Loaded CLIP model: {model_name}")
        
        # 创建图像预处理流程
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # OpenCV转PIL
            transforms.Resize((224, 224)),  # CLIP要求224x224输入
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
        
        self.normalize = normalize  # 是否归一化特征向量

    def process_image(self, image):
        """处理不同格式的输入图像，返回OpenCV格式"""
        # 处理文件路径输入
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法加载图像: {image}")
            return img
        
        # 处理numpy数组输入 (OpenCV格式)
        if isinstance(image, np.ndarray):
            # 确保是三维数组 (HxWxC)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("无效的图像尺寸，期望HxWx3格式")
            return image
        
        # 处理PIL图像输入
        #if isinstance(image, Image.Image):
         #   return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        raise TypeError("不支持的图像类型，支持: str, np.ndarray, PIL.Image")

    def extract(self, image):
        """从图像中提取512维CLIP特征"""
        try:
            # 转换输入图像为OpenCV格式 (BGR)
            cv_image = self.process_image(image)
            
            # 检查图像尺寸
            h, w, c = cv_image.shape
            if not (h == 720 and w == 1280 and c == 3):
                print(f"警告: 图像尺寸为{h}x{w}x{c}，但期望720x1280x3")
            
            # 应用预处理
            image_tensor = self.transform(cv_image)  # 形状: [3, 224, 224]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # 添加批维度 [1, 3, 224, 224]
            
            # 提取特征
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)  # 形状: [1, 512]
            
            # 转换为numpy向量
            features = features.cpu().numpy().flatten()  # 形状: (512,)
            
            # 可选归一化
            if self.normalize:
                features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
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
        
        # 展平特征向量
        features1 = features1.flatten()
        features2 = features2.flatten()
        
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
            distance = 1.0 - cosine_sim
        
        elif distance_type == 'euclidean':
            # 欧氏距离
            distance = np.linalg.norm(features1 - features2)
        
        elif distance_type == 'manhattan':
            # 曼哈顿距离
            distance = np.sum(np.abs(features1 - features2))
        
        else:
            rospy.logwarn(f"Unknown distance type: {distance_type}. Using cosine.")
            return self.distance(features1, features2, 'cosine')
        
        # 确保返回标量
        if isinstance(distance, np.ndarray):
            return distance.item()
        return distance 


from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image
import numpy as np
import os
import rospy
import cv2
import time
from sklearn.neighbors import KDTree
class VNMTopoMapper:
    def __init__(self):
        rospy.init_node('vnm_topo_mapper')
        
        # 参数配置
        self.map_dir = rospy.get_param('~map_dir', '~/maps')
        self.map_name = rospy.get_param('~map_name', 'office_map')
        self.min_node_distance = rospy.get_param('~min_node_distance', 0.5)  # 米
        self.max_angular = rospy.get_param('~max_angular', 0.5)  # 弧度
        self.connect_radius = rospy.get_param('~connect_radius', 3.0)  # 连接半径（米）
        self.feature_threshold = rospy.get_param('~feature_threshold', 0.35)  # 特征距离阈值
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        
        # 创建输出目录
        self.map_path = os.path.join(os.path.expanduser(self.map_dir), self.map_name)
        os.makedirs(self.map_path, exist_ok=True)
        os.makedirs(os.path.join(self.map_path, "images"), exist_ok=True)
        
        # 初始化组件
        self.bridge = CvBridge()
        self.topo_graph = TopoGraph()
        self.feature_extractor = FeatureExtractor()
        
        # 节点位置索引
        self.node_positions = []  # 存储所有节点位置 (x, y)
        self.position_tree = None  # KDTree用于空间搜索
        
        # 状态变量
        self.last_node_position = None  # 上一个节点的完整位置 (x, y, theta)
        self.last_node_yaw = None
        self.exploring = True
        self.current_image = None
        self.current_position = None  # 当前位置 (x, y)
        self.current_yaw = None  # 当前朝向
        
        # ROS接口
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        
        # 服务
        self.start_service = rospy.Service('start_exploration', Empty, self.start_exploration)
        self.stop_service = rospy.Service('stop_exploration', Empty, self.stop_exploration)
        self.save_service = rospy.Service('save_map', Empty, self.save_map)
        
        # 初始化定时器 - 这才是探索循环的核心
        self.exploration_timer = None
        
        rospy.loginfo("VNM TopoMapper initialized. Ready to explore.")
        rospy.on_shutdown(self.shutdown)
    
    def image_callback(self, msg):
        """图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            rospy.logerr(f"图像转换错误: {e}")
    
    def odom_callback(self, msg):
        """里程计回调函数"""
        try:
            self.current_pose = msg.pose.pose
            self.current_position = [
                self.current_pose.position.x,
                self.current_pose.position.y
            ]
            
            # 计算偏航角
            orientation = self.current_pose.orientation
            quat = (orientation.x, orientation.y, orientation.z, orientation.w)
            _, _, yaw = euler_from_quaternion(quat)
            self.current_yaw = yaw
        except Exception as e:
            rospy.logerr(f"里程计处理错误: {e}")
    
    def start_exploration(self, req):
        """开始探索服务"""
        #if not self.exploring:
        self.exploring = True
            # 创建新节点
        if self.current_image is not None and self.current_position is not None:
                self.create_new_node()
            
            # 启动探索定时器
        if self.exploration_timer is None:
                self.exploration_timer = rospy.Timer(rospy.Duration(0.1), self.exploration_loop_callback)
        rospy.loginfo("开始环境探索")
        return EmptyResponse()
    
    def stop_exploration(self, req):
        """停止探索服务"""
        if self.exploring:
            self.exploring = False
            if self.exploration_timer is not None:
                self.exploration_timer.shutdown()
                self.exploration_timer = None
            rospy.loginfo("停止环境探索")
        return EmptyResponse()
    
    def exploration_loop_callback(self, event):
        """探索循环回调 - 定时器触发时执行"""
        if not self.exploring:
            return
        
        try:
            self.exploration_loop()
        except Exception as e:
            rospy.logerr(f"探索循环错误: {e}")
    
    def exploration_loop(self):
        """主探索逻辑"""
        # 检查必要数据
        if self.current_image is None:
            rospy.logwarn("当前无图像数据，等待图像输入")
            return
            
        if self.current_position is None:
            rospy.logwarn("当前无位置数据，等待里程计输入")
            return
            
        # 第一次访问，创建起始节点
        if self.last_node_position is None:
            self.create_new_node()
            return
            
        # 计算距离和角度差
        position_change = np.linalg.norm(
            np.array(self.current_position) - 
            np.array(self.last_node_position[:2])
        )
        yaw_change = abs(self.current_yaw - self.last_node_yaw)
        
        # 创建新节点的条件
        if position_change > self.min_node_distance or yaw_change > self.max_angular:
            self.create_new_node()
    def save_map(self, req):
        """保存地图服务"""
        try:
            # 保存拓扑图对象
            graph_file = os.path.join(self.map_path, f"{self.map_name}.graph")
            self.topo_graph.save(graph_file)
            rospy.loginfo(f"拓扑图保存至: {graph_file}")
            
            # 导出GraphML格式用于可视化
            graphml_file = os.path.join(self.map_path, f"{self.map_name}.graphml")
            self.export_graphml(graphml_file)
            rospy.loginfo(f"GraphML可视化文件保存至: {graphml_file}")
        except Exception as e:
            rospy.logerr(f"保存地图失败: {e}")
        return EmptyResponse()
    
    
    def export_graphml(self, filename):
        """导出GraphML格式的可视化文件"""
        try:
            nodes_data = {}
            
            # 准备节点数据
            for node_id, node in self.topo_graph.nodes.items():
                # 添加节点属性
                nodes_data[node_id] = {
                    'x': float(node.position[0]),
                    'y': float(node.position[1]),
                    'theta': float(node.position[2])
                }
                
                # 保存节点图像
                img_path = os.path.join(self.map_path, "images", f"node_{node_id}.jpg")
                cv2.imwrite(img_path, node.image)
            
            # 创建NetworkX图
            G = nx.Graph()
            for node_id, data in nodes_data.items():
                G.add_node(node_id, **data)
            
            # 添加边
            for edge in self.topo_graph.graph.edges(data=True):
                G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
            
            # 保存为GraphML格式
            nx.write_graphml(G, filename)
        except Exception as e:
            rospy.logerr(f"导出GraphML失败: {e}")
    
    def update_position_tree(self):
        """更新位置KDTree索引"""
        try:
            if not self.node_positions:
                return
                
            # 将位置列表转换为NumPy数组
            positions_array = np.array(self.node_positions)
            
            # 创建或更新KDTree
            self.position_tree = KDTree(positions_array)
            rospy.logdebug("位置KDTree索引已更新")
        except Exception as e:
            rospy.logwarn(f"更新位置索引失败: {e}")
    
    def find_spatial_neighbors(self, position):
        """查找空间距离内的邻居节点"""
        if self.position_tree is None or len(self.node_positions) == 0:
            return []
            
        try:    
            # 使用KDTree查询半径内的节点
            position_2d = np.array([position[0], position[1]])
            neighbor_indices = self.position_tree.query_radius([position_2d], r=self.connect_radius)[0]
            
            return neighbor_indices
        except Exception as e:
            rospy.logwarn(f"查找空间邻居失败: {e}")
            return []
    
    def create_new_node(self):
        """创建新的拓扑节点并连接邻居"""
        try:
            # 提取图像特征
            features = self.feature_extractor.extract(self.current_image)
            if features is None:
                rospy.logwarn("特征提取失败，跳过节点创建")
                return
                
            # 构建完整位置信息 (x, y, theta)
            position = [self.current_position[0], self.current_position[1], self.current_yaw]
            
            # 添加到拓扑图
            node_id = self.topo_graph.add_node(
                image=self.current_image,
                position=position,
                features=features
            )
            
            rospy.loginfo(f"创建拓扑节点 #{node_id} 位置 ({position[0]:.2f}, {position[1]:.2f})")
            
            # 记录状态
            self.last_node_position = position
            self.last_node_yaw = self.current_yaw
            
            # 更新位置索引
            self.node_positions.append(position[:2])  # 仅存储x,y
            self.update_position_tree()
            
            # 连接到邻近节点（仅当不是第一个节点时）
            if node_id > 0:
                self.connect_to_neighbors(node_id, position, features)
        except Exception as e:
            rospy.logerr(f"创建节点失败: {e}")
    
    def connect_to_neighbors(self, new_node_id, new_position, new_features):
        """连接新节点到空间和特征相似的邻居"""
        try:
            # 1. 强制连接到前一个节点（保证轨迹连续性）
            prev_node_id = new_node_id - 1
            if prev_node_id in self.topo_graph.nodes:
                prev_features = self.topo_graph.nodes[prev_node_id].features
                
                # 确保特征向量是一维数组
                prev_features = prev_features.flatten()
                new_features = new_features.flatten()
                
                feat_dist = self.feature_extractor.distance(new_features, prev_features)
                
                # 确保距离是标量
                if isinstance(feat_dist, np.ndarray):
                    feat_dist = feat_dist.item()  # 转换为标量
                    
                self.topo_graph.add_edge(new_node_id, prev_node_id, feat_dist)
                rospy.loginfo(f"强制连接节点 {new_node_id} -> {prev_node_id}, 距离: {feat_dist:.4f}")
            else:
                rospy.logwarn(f"无法连接前节点, {prev_node_id} 不存在")
            
            # 2. 查找空间邻近节点
            spatial_neighbors = self.find_spatial_neighbors(new_position)
            rospy.logdebug(f"为节点 {new_node_id} 找到 {len(spatial_neighbors)} 个空间邻居")
            
            # 3. 连接到空间邻近节点（特征相似时）
            connected_to_neighbor = False
            for neighbor_id in spatial_neighbors:
                if neighbor_id == new_node_id:  # 跳过自身
                    continue
                    
                # 跳过刚刚强制连接的节点
                if neighbor_id == prev_node_id:
                    continue
                    
                neighbor_node = self.topo_graph.nodes[neighbor_id]
                
                # 确保特征向量是一维数组
                neighbor_features = neighbor_node.features.flatten()
                new_features_flat = new_features.flatten()
                
                # 计算特征距离
                feat_dist = self.feature_extractor.distance(
                    new_features_flat, neighbor_features
                )
                
                # 确保距离是标量
                if isinstance(feat_dist, np.ndarray):
                    feat_dist = feat_dist.item()  # 转换为标量
                    
                # 如果特征距离小于阈值则创建连接
                if feat_dist < self.feature_threshold:
                    # 添加双向连接
                    self.topo_graph.add_edge(new_node_id, neighbor_id, feat_dist)
                    connected_to_neighbor = True
                    rospy.loginfo(f"添加连接 {new_node_id} <-> {neighbor_id}, 特征距离: {feat_dist:.4f}")
                    
            # 4. 回退：如果没有连接，保证连接到最近的空间节点
            if not connected_to_neighbor and spatial_neighbors:
                # 查找最近的邻居
                min_spatial_dist = float('inf')
                nearest_neighbor = None
                
                for neighbor_id in spatial_neighbors:
                    if neighbor_id == new_node_id or neighbor_id == prev_node_id:
                        continue
                        
                    neighbor_pos = self.topo_graph.nodes[neighbor_id].position[:2]
                    spatial_dist = np.linalg.norm(np.array(new_position[:2]) - np.array(neighbor_pos))
                    
                    if spatial_dist < min_spatial_dist:
                        min_spatial_dist = spatial_dist
                        nearest_neighbor = neighbor_id
                
                # 添加连接
                if nearest_neighbor is not None:
                    self.topo_graph.add_edge(new_node_id, nearest_neighbor, min_spatial_dist)
                    rospy.loginfo(f"添加回退连接 {new_node_id} -> {nearest_neighbor}, 空间距离: {min_spatial_dist:.2f}m")
        except Exception as e:
            rospy.logerr(f"连接邻居节点失败: {e}")
    
    def shutdown(self):
        """节点关闭时的清理工作"""
        rospy.loginfo("关闭拓扑地图构建器...")
        try:
            if self.exploration_timer is not None:
                self.exploration_timer.shutdown()
            self.save_map(EmptyRequest())
        except:
            pass
        rospy.loginfo("拓扑地图构建器已关闭")

'''def main():
    try:
        mapper = VNMTopoMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"拓扑地图构建器发生严重错误: {e}")

if __name__ == '__main__':
    main()'''



if __name__ == '__main__':
    try:
        mapper = VNMTopoMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"拓扑地图构建器发生严重错误: {e}")