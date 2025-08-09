#!/usr/bin/env python3
import rospy
#from llm import WordList
from word import WordSplitter
import os
from llm.msg import WordList
class WordSplitterNode:
    def __init__(self):
        # 获取模型路径参数
        model_path = rospy.get_param('~model_path', '/home/rosnoetic/nav02_ws/src/flan-t5-base')
        
        # 初始化模型
        self.splitter = WordSplitter(model_path)
        
        # 创建发布器
        self.pub = rospy.Publisher('word_list', WordList, queue_size=10)
        
        # 获取输入句子参数
        sentence = rospy.get_param('~sentence', '')
        
        if sentence:
            rospy.loginfo(f"处理句子: '{sentence}'")
            self.process_and_publish(sentence)
        else:
            rospy.logwarn("未提供句子参数，节点将退出")
            rospy.signal_shutdown("未提供输入句子")
    
    def process_and_publish(self, sentence):
        """处理句子并发布结果"""
        try:
            # 拆分句子
            words = self.splitter.split_sentence(sentence)
            
            # 创建消息
            msg = WordList()
            msg.words = words
            
            # 发布消息
            self.pub.publish(msg)
            rospy.loginfo(f"发布词语列表: {words}")
            
        except Exception as e:
            rospy.logerr(f"处理失败: {str(e)}")

if __name__ == '__main__':
    rospy.init_node('word_splitter_node')
    try:
        node = WordSplitterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass