(require 'dash)

(defun double (x) (+ x x))
(defun quadruple (x) (double (double x)))
(defun factorial (n) (-product (number-sequence 1 n)))
(defun average (ns) (/ (-sum ns) (length ns)))
