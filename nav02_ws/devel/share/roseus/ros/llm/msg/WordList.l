;; Auto-generated. Do not edit!


(when (boundp 'llm::WordList)
  (if (not (find-package "LLM"))
    (make-package "LLM"))
  (shadow 'WordList (find-package "LLM")))
(unless (find-package "LLM::WORDLIST")
  (make-package "LLM::WORDLIST"))

(in-package "ROS")
;;//! \htmlinclude WordList.msg.html


(defclass llm::WordList
  :super ros::object
  :slots (_words ))

(defmethod llm::WordList
  (:init
   (&key
    ((:words __words) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _words __words)
   self)
  (:words
   (&optional __words)
   (if __words (setq _words __words)) _words)
  (:serialization-length
   ()
   (+
    ;; string[] _words
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _words)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _words
     (write-long (length _words) s)
     (dolist (elem _words)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _words
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _words (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _words i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(setf (get llm::WordList :md5sum-) "2bd0e7dd008cf8f52a5113ba090403b7")
(setf (get llm::WordList :datatype-) "llm/WordList")
(setf (get llm::WordList :definition-)
      "string[] words
")



(provide :llm/WordList "2bd0e7dd008cf8f52a5113ba090403b7")


