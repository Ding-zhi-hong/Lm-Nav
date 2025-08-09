;; Auto-generated. Do not edit!


(when (boundp 'topograph::ExecutePath)
  (if (not (find-package "TOPOGRAPH"))
    (make-package "TOPOGRAPH"))
  (shadow 'ExecutePath (find-package "TOPOGRAPH")))
(unless (find-package "TOPOGRAPH::EXECUTEPATH")
  (make-package "TOPOGRAPH::EXECUTEPATH"))
(unless (find-package "TOPOGRAPH::EXECUTEPATHREQUEST")
  (make-package "TOPOGRAPH::EXECUTEPATHREQUEST"))
(unless (find-package "TOPOGRAPH::EXECUTEPATHRESPONSE")
  (make-package "TOPOGRAPH::EXECUTEPATHRESPONSE"))

(in-package "ROS")





(defclass topograph::ExecutePathRequest
  :super ros::object
  :slots ())

(defmethod topograph::ExecutePathRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass topograph::ExecutePathResponse
  :super ros::object
  :slots (_success ))

(defmethod topograph::ExecutePathResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass topograph::ExecutePath
  :super ros::object
  :slots ())

(setf (get topograph::ExecutePath :md5sum-) "358e233cde0c8a8bcfea4ce193f8fc15")
(setf (get topograph::ExecutePath :datatype-) "topograph/ExecutePath")
(setf (get topograph::ExecutePath :request) topograph::ExecutePathRequest)
(setf (get topograph::ExecutePath :response) topograph::ExecutePathResponse)

(defmethod topograph::ExecutePathRequest
  (:response () (instance topograph::ExecutePathResponse :init)))

(setf (get topograph::ExecutePathRequest :md5sum-) "358e233cde0c8a8bcfea4ce193f8fc15")
(setf (get topograph::ExecutePathRequest :datatype-) "topograph/ExecutePathRequest")
(setf (get topograph::ExecutePathRequest :definition-)
      "# 执行路径请求（空请求）
---
# 执行路径响应
bool success                     # 是否成功执行
")

(setf (get topograph::ExecutePathResponse :md5sum-) "358e233cde0c8a8bcfea4ce193f8fc15")
(setf (get topograph::ExecutePathResponse :datatype-) "topograph/ExecutePathResponse")
(setf (get topograph::ExecutePathResponse :definition-)
      "# 执行路径请求（空请求）
---
# 执行路径响应
bool success                     # 是否成功执行
")



(provide :topograph/ExecutePath "358e233cde0c8a8bcfea4ce193f8fc15")


