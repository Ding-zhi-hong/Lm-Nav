;; Auto-generated. Do not edit!


(when (boundp 'topograph::PlanPath)
  (if (not (find-package "TOPOGRAPH"))
    (make-package "TOPOGRAPH"))
  (shadow 'PlanPath (find-package "TOPOGRAPH")))
(unless (find-package "TOPOGRAPH::PLANPATH")
  (make-package "TOPOGRAPH::PLANPATH"))
(unless (find-package "TOPOGRAPH::PLANPATHREQUEST")
  (make-package "TOPOGRAPH::PLANPATHREQUEST"))
(unless (find-package "TOPOGRAPH::PLANPATHRESPONSE")
  (make-package "TOPOGRAPH::PLANPATHRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "NAV_MSGS"))
  (ros::roseus-add-msgs "nav_msgs"))


(defclass topograph::PlanPathRequest
  :super ros::object
  :slots (_start_pose _landmarks ))

(defmethod topograph::PlanPathRequest
  (:init
   (&key
    ((:start_pose __start_pose) (instance geometry_msgs::Pose :init))
    ((:landmarks __landmarks) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _start_pose __start_pose)
   (setq _landmarks __landmarks)
   self)
  (:start_pose
   (&rest __start_pose)
   (if (keywordp (car __start_pose))
       (send* _start_pose __start_pose)
     (progn
       (if __start_pose (setq _start_pose (car __start_pose)))
       _start_pose)))
  (:landmarks
   (&optional __landmarks)
   (if __landmarks (setq _landmarks __landmarks)) _landmarks)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _start_pose
    (send _start_pose :serialization-length)
    ;; string[] _landmarks
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _landmarks)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _start_pose
       (send _start_pose :serialize s)
     ;; string[] _landmarks
     (write-long (length _landmarks) s)
     (dolist (elem _landmarks)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _start_pose
     (send _start_pose :deserialize buf ptr-) (incf ptr- (send _start_pose :serialization-length))
   ;; string[] _landmarks
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _landmarks (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _landmarks i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(defclass topograph::PlanPathResponse
  :super ros::object
  :slots (_path ))

(defmethod topograph::PlanPathResponse
  (:init
   (&key
    ((:path __path) (instance nav_msgs::Path :init))
    )
   (send-super :init)
   (setq _path __path)
   self)
  (:path
   (&rest __path)
   (if (keywordp (car __path))
       (send* _path __path)
     (progn
       (if __path (setq _path (car __path)))
       _path)))
  (:serialization-length
   ()
   (+
    ;; nav_msgs/Path _path
    (send _path :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; nav_msgs/Path _path
       (send _path :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; nav_msgs/Path _path
     (send _path :deserialize buf ptr-) (incf ptr- (send _path :serialization-length))
   ;;
   self)
  )

(defclass topograph::PlanPath
  :super ros::object
  :slots ())

(setf (get topograph::PlanPath :md5sum-) "ac039c3aab9885e2d76b26c6b9340c81")
(setf (get topograph::PlanPath :datatype-) "topograph/PlanPath")
(setf (get topograph::PlanPath :request) topograph::PlanPathRequest)
(setf (get topograph::PlanPath :response) topograph::PlanPathResponse)

(defmethod topograph::PlanPathRequest
  (:response () (instance topograph::PlanPathResponse :init)))

(setf (get topograph::PlanPathRequest :md5sum-) "ac039c3aab9885e2d76b26c6b9340c81")
(setf (get topograph::PlanPathRequest :datatype-) "topograph/PlanPathRequest")
(setf (get topograph::PlanPathRequest :definition-)
      "geometry_msgs/Pose start_pose  
string[] landmarks   


================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---

nav_msgs/Path path           

================================================================================
MSG: nav_msgs/Path
#An array of poses that represents a Path for a robot to follow
Header header
geometry_msgs/PoseStamped[] poses

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/PoseStamped
# A Pose with reference coordinate frame and timestamp
Header header
Pose pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")

(setf (get topograph::PlanPathResponse :md5sum-) "ac039c3aab9885e2d76b26c6b9340c81")
(setf (get topograph::PlanPathResponse :datatype-) "topograph/PlanPathResponse")
(setf (get topograph::PlanPathResponse :definition-)
      "geometry_msgs/Pose start_pose  
string[] landmarks   


================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---

nav_msgs/Path path           

================================================================================
MSG: nav_msgs/Path
#An array of poses that represents a Path for a robot to follow
Header header
geometry_msgs/PoseStamped[] poses

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/PoseStamped
# A Pose with reference coordinate frame and timestamp
Header header
Pose pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")



(provide :topograph/PlanPath "ac039c3aab9885e2d76b26c6b9340c81")


