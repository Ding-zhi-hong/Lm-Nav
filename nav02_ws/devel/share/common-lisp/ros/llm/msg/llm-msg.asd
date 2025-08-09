
(cl:in-package :asdf)

(defsystem "llm-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "WordList" :depends-on ("_package_WordList"))
    (:file "_package_WordList" :depends-on ("_package"))
  ))