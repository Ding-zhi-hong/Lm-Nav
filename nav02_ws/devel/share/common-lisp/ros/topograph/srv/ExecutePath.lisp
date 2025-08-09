; Auto-generated. Do not edit!


(cl:in-package topograph-srv)


;//! \htmlinclude ExecutePath-request.msg.html

(cl:defclass <ExecutePath-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass ExecutePath-request (<ExecutePath-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ExecutePath-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ExecutePath-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name topograph-srv:<ExecutePath-request> is deprecated: use topograph-srv:ExecutePath-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ExecutePath-request>) ostream)
  "Serializes a message object of type '<ExecutePath-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ExecutePath-request>) istream)
  "Deserializes a message object of type '<ExecutePath-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ExecutePath-request>)))
  "Returns string type for a service object of type '<ExecutePath-request>"
  "topograph/ExecutePathRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ExecutePath-request)))
  "Returns string type for a service object of type 'ExecutePath-request"
  "topograph/ExecutePathRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ExecutePath-request>)))
  "Returns md5sum for a message object of type '<ExecutePath-request>"
  "358e233cde0c8a8bcfea4ce193f8fc15")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ExecutePath-request)))
  "Returns md5sum for a message object of type 'ExecutePath-request"
  "358e233cde0c8a8bcfea4ce193f8fc15")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ExecutePath-request>)))
  "Returns full string definition for message of type '<ExecutePath-request>"
  (cl:format cl:nil "# 执行路径请求（空请求）~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ExecutePath-request)))
  "Returns full string definition for message of type 'ExecutePath-request"
  (cl:format cl:nil "# 执行路径请求（空请求）~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ExecutePath-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ExecutePath-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ExecutePath-request
))
;//! \htmlinclude ExecutePath-response.msg.html

(cl:defclass <ExecutePath-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass ExecutePath-response (<ExecutePath-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ExecutePath-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ExecutePath-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name topograph-srv:<ExecutePath-response> is deprecated: use topograph-srv:ExecutePath-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <ExecutePath-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader topograph-srv:success-val is deprecated.  Use topograph-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ExecutePath-response>) ostream)
  "Serializes a message object of type '<ExecutePath-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ExecutePath-response>) istream)
  "Deserializes a message object of type '<ExecutePath-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ExecutePath-response>)))
  "Returns string type for a service object of type '<ExecutePath-response>"
  "topograph/ExecutePathResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ExecutePath-response)))
  "Returns string type for a service object of type 'ExecutePath-response"
  "topograph/ExecutePathResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ExecutePath-response>)))
  "Returns md5sum for a message object of type '<ExecutePath-response>"
  "358e233cde0c8a8bcfea4ce193f8fc15")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ExecutePath-response)))
  "Returns md5sum for a message object of type 'ExecutePath-response"
  "358e233cde0c8a8bcfea4ce193f8fc15")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ExecutePath-response>)))
  "Returns full string definition for message of type '<ExecutePath-response>"
  (cl:format cl:nil "# 执行路径响应~%bool success                     # 是否成功执行~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ExecutePath-response)))
  "Returns full string definition for message of type 'ExecutePath-response"
  (cl:format cl:nil "# 执行路径响应~%bool success                     # 是否成功执行~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ExecutePath-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ExecutePath-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ExecutePath-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ExecutePath)))
  'ExecutePath-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ExecutePath)))
  'ExecutePath-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ExecutePath)))
  "Returns string type for a service object of type '<ExecutePath>"
  "topograph/ExecutePath")