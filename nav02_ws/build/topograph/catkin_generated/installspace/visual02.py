#!/usr/bin/env python3
import numpy as np
import networkx as nx
import os
import cv2
import pickle
import rospy
import time

# 非ROS环境下的rospy.Time.now()替代方案
try:
    rospy.Time.now()
except (rospy.exceptions.ROSInitException, NameError):
    class MockRosTime:
        @staticmethod
        def now():
            return time.time()
    rospy.Time = MockRosTime

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
        self.graph = nx.Graph()
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
        # 自定义Unpickler处理类映射
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # 重定向到当前模块中定义的类
                if name == 'TopoNode':
                    return TopoNode
                if name == 'TopoGraph':
                    return TopoGraph
                return super().find_class(module, name)
        
        with open(filename, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
            self.graph = data['graph']
            self.nodes = data['nodes']
            self.node_counter = data['node_counter']

# 以下是可视化的主代码
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 创建并加载拓扑图
    topo_graph = TopoGraph()
    map_file = "/home/rosnoetic/maps/office_map/office_map.graph"
    topo_graph.load(map_file)
    
    # 从TopoGraph对象中获取图数据和节点
    G = topo_graph.graph
    nodes = topo_graph.nodes
    
    # 打印基本信息
    print(f"成功加载拓扑图，包含 {len(topo_graph.nodes)} 个节点")
    print(f"节点计数器: {topo_graph.node_counter}")
    
    # 设置画布
    plt.figure(figsize=(12, 10))
    
    # 创建位置字典 (使用节点存储的位置)
    pos = {}
    node_colors = []
    node_sizes = []
    labels = {}
    
    # 收集节点位置并设置可视化属性
    for node_id, node in nodes.items():
        # 使用x,y位置（忽略theta）
        pos[node_id] = (node.position[0], node.position[1])
        
        # 根据节点ID设置颜色（使用HSV色轮循环）
        hue = (node_id * 0.618) % 1.0  # 黄金比例创建均匀分布的颜色
        node_colors.append((hue, 0.7, 0.9))  # HSV值转换为RGB
        
        # 根据连接数设置节点大小
        degree = len(list(G.neighbors(node_id)))
        node_sizes.append(300 + 50 * degree)
        
        # 每5个节点添加一个标签，避免过多标签造成的混乱
        if node_id % 5 == 0:
            labels[node_id] = f"{node_id}"
    
    # 将HSV颜色转换为RGB
    from matplotlib.colors import hsv_to_rgb
    node_colors_rgb = [hsv_to_rgb(c) for c in node_colors]
    
    # 绘制拓扑图
    plt.title('Office Topological Map Visualization', fontsize=16)
    plt.grid(True, alpha=0.2)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    
    # 绘制节点、边和标签
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors_rgb, 
        alpha=0.9,
        edgecolors='black'
    )
    
    nx.draw_networkx_edges(
        G, pos, 
        width=1.5, 
        edge_color='blue', 
        alpha=0.6,
        connectionstyle="arc3,rad=0.1"
    )
    
    nx.draw_networkx_labels(
        G, pos, 
        labels, 
        font_size=10, 
        font_weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # 添加图例说明
    plt.figtext(
        0.5, 0.01, 
        f"Topological Map: {len(nodes)} nodes, {G.number_of_edges()} edges",
        ha="center", fontsize=10
    )
    
    # 保存和显示图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为图例留出空间
    output_path = "/home/rosnoetic/maps/office_map/office_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图像已保存至: {output_path}")
    
    plt.show()