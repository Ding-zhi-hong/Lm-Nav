#!/usr/bin/env python3
import rospy
import numpy as np
import networkx as nx
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from topograph.srv import PlanPath, PlanPathResponse
import dill
import os
import cv2
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

class PathPlanner:
    def __init__(self):
        rospy.init_node('path_planner')
        
        # 加载拓扑图
        map_file = "/home/rosnoetic/maps/office_map/office_map_1754449140.graph"
        self.topo_graph = TopoGraph()
        self.topo_graph.load(map_file)
        rospy.loginfo(f"Loaded topology graph with {len(self.topo_graph.nodes)} nodes")
        
        # ROS接口
        self.path_pub = rospy.Publisher('/nav_path', Path, queue_size=1)
        self.service = rospy.Service('plan_path', PlanPath, self.handle_planning)
        
        # 规划参数
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.2)  # 米
        self.landmark_weight = rospy.get_param('~landmark_weight', 0.7)  # 地标概率权重
        
        rospy.loginfo("Path Planner initialized")
    
    def handle_planning(self, req):
        """处理路径规划请求"""
        rospy.loginfo(f"Planning path for landmarks: {req.landmarks}")
        
        # 获取起点节点
        start_node = self.find_nearest_node(req.start_pose)
        rospy.loginfo(f"Starting from node {start_node}")
        
        # 分阶段规划路径
        full_path = []
        current_node = start_node
        
        for landmark in req.landmarks:
            # 找到该地标概率最高的节点
            end_node = self.find_highest_prob_node(landmark)
            if end_node is None:
                rospy.logwarn(f"No high-probability node found for landmark: {landmark}")
                continue
                
            rospy.loginfo(f"Navigating to landmark '{landmark}' at node {end_node}")
            
            # 规划路径段
            path_segment = self.plan_path_segment(current_node, end_node, landmark)
            full_path.extend(path_segment)
            
            # 更新当前节点
            current_node = end_node
        
        # 创建Path消息
        nav_path = self.create_path_message(full_path)
        self.path_pub.publish(nav_path)
        
        return PlanPathResponse(path=nav_path)
    
    def find_nearest_node(self, pose):
        """根据位置找到最近节点"""
        position = [pose.position.x, pose.position.y]
        return self.topo_graph.find_nearest_node(position)[0]
    
    def find_highest_prob_node(self, landmark):
        """找到指定地标概率最高的节点"""
        max_prob = -1
        best_node = None
        
        for node_id, node in self.topo_graph.nodes.items():
            if landmark in node.landmark_probs:
                prob = node.landmark_probs[landmark]
                if prob > max_prob:
                    max_prob = prob
                    best_node = node_id
        
        return best_node
    
    def plan_path_segment(self, start, goal, landmark):
        """规划两点之间的路径"""
        # 自定义启发式函数，考虑地标概率
        def heuristic(n1, n2):
            # 位置距离部分
            pos1 = self.topo_graph.nodes[n1].position
            pos2 = self.topo_graph.nodes[n2].position
            dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
            
            # 地标概率部分
            if n2 in self.topo_graph.nodes and landmark in self.topo_graph.nodes[n2].landmark_probs:
                prob_factor = 1.0 - self.topo_graph.nodes[n2].landmark_probs[landmark]
            else:
                prob_factor = 1.0
            
            # 加权组合
            return dist * (self.landmark_weight * prob_factor + (1 - self.landmark_weight))
        
        # 使用A*算法找到最短路径
        try:
            path = nx.astar_path(
                self.topo_graph.graph,
                start,
                goal,
                heuristic=heuristic,
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            rospy.logwarn(f"No path found from {start} to {goal}")
            return []
    
    def create_path_message(self, node_path):
        """创建ROS Path消息"""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for node_id in node_path:
            node = self.topo_graph.nodes[node_id]
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "map"
            pose.pose.position.x = node.position[0]
            pose.pose.position.y = node.position[1]
            pose.pose.position.z = 0.0
            
            # 设置方向（使用节点存储的偏航角）
            from tf.transformations import quaternion_from_euler
            q = quaternion_from_euler(0, 0, node.position[2])
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            path_msg.poses.append(pose)
        
        rospy.loginfo(f"Created path with {len(path_msg.poses)} waypoints")
        return path_msg

if __name__ == '__main__':
    planner = PathPlanner()
    rospy.spin()