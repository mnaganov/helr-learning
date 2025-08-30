# 6. Recursive functions

## 6.1 Basic concepts

`chapter6-functions.hs`
```haskell
fac :: Int -> Int
fac 0 = 1
fac n = n * fac(n-1)
```

```haskell
λ> fac 3
6
λ> fac 10
3628800
```

```elisp
(defun fac(n)
  (cond ((= n 0) 1)
        (t (* n (fac (1- n))))))
(fac 3)
6
(fac 10)
3628800
```

**Note:** Don't forget that Elisp lacks tail recursion elimination.

```rust
>> fn fac(n: i32) -> i32 { match n { 0 => 1, _ => n * fac(n-1) } }
>> fac(3)
6
>> fac(10)
3628800
```

## 6.2 Recursion on lists

`chapter6-functions.hs`
```haskell
prodct :: Num a => [a] -> a
prodct []     = 1
prodct (n:ns) = n * prodct ns

lngth :: [a] -> Int
lngth [] = 0
lngth (_:xs) = 1 + lngth xs

revrse :: [a] -> [a]
revrse [] = []
revrse (x:xs) = revrse xs ++ [x]

insert :: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x (y:ys) | x <= y = x : y : ys
                | otherwise = y : insert x ys

isort :: Ord a => [a] -> [a]
isort [] = []
isort (x:xs) = insert x (isort xs)
```

```haskell
λ> prodct [2,3,4]
24
λ> lngth [1,2,3,4]
4
λ> revrse [1,2,3]
[3,2,1]
λ> insert 3 [1,2,4,5]
[1,2,3,4,5]
λ> isort [3,2,1,4]
[1,2,3,4]
```

**Note:** To simplify code we have slightly renamed re-implementations
of the functions from the standard prelude. This way, we don't have to
prefix them with `Main.` on every use.

```elisp
(defun product(l)
  (pcase l
    ('nil 1)
    (`(,n . ,ns) (* n (product ns)))))
(product '(2 3 4))
24

(defun lngth(l)
  (pcase l
    ('nil 0)
    (`(,n . ,ns) (+ 1 (lngth ns)))))
(lngth '(1 2 3 4))
4

(defun revrse(l r)
  (pcase l
    ('nil r)
    (`(,x . ,xs) (revrse xs (cons x r)))))
(revrse '(1 2 3) nil)
(3 2 1)
```

**Note:** Since appending an element to the tail of a list is inefficient—
it requires re-creating every cons cell of it, the implementation of `revrse`
"conses" on the head instead. This requires passing the resulting list as the
second parameter which initially must be an empty list.

```elisp
(defun insrt(x l)
  (pcase l
    ('nil (list x))
    (`(,y . ,ys) (if (<= x y) (cons x l) (cons y (insrt x ys))))))
(insrt 3 nil)
(3)
(insrt 3 '(1 2 4 5))
(1 2 3 4 5)

(defun isort(l)
  (pcase l
    ('nil nil)
    (`(,x . ,xs) (insrt x (isort xs)))))
(isort '(3 2 1 4))
(1 2 3 4)
```

```rust
>> fn product(l: &[i32]) -> i32 { match l { [] => 1, [n, ns @ ..] => n * product(ns) } }
>> product(&[2,3,4])
24

>> fn lngth<T>(l: &[T]) -> usize { match l { [] => 0, [_, xs @ ..] => 1 + lngth(xs) } }
>> lngth(&[1,2,3,4])
4

>> fn reverse<T: Clone>(l: Vec<T>) -> Vec<T> { if l.is_empty() { l } else { let mut v = l.to_vec(); let mut xs = reverse::<T>(v.split_off(1)); xs.extend(v); xs } }
>> reverse(vec![1,2,3])
[3, 2, 1]

>> fn insert(x: i32, l: Vec<i32>) -> Vec<i32> { let mut r = vec![x]; if l.is_empty() { r } else {let mut y = l.to_vec(); let ys = y.split_off(1); if x <= y[0] { r.extend(y); r.extend(ys); r } else { y.extend(insert(x, ys)); y } } }
>> insert(3, vec![1,2,4,5])
[1, 2, 3, 4, 5]

>> fn isort(l: Vec<i32>) -> Vec<i32> { if l.is_empty() { l } else { let mut x = l.to_vec(); let xs = x.split_off(1); insert(x[0], isort(xs)) } }
>> isort(vec![3,2,1,4])
[1, 2, 3, 4]
```

## 6.3 Multiple arguments

`chapter6-functions.hs`
```haskell
zp :: [a] -> [b] -> [(a,b)]
zp [] _          = []
zp _ []          = []
zp (x:xs) (y:ys) = (x,y) : zp xs ys

drp :: Int -> [a] -> [a]
drp 0 xs     = xs
drp _ []     = []
drp n (_:xs) = drp (n-1) xs
```

```haskell
λ> zp ['a','b','c'] [1,2,3,4]
[('a',1),('b',2),('c',3)]
λ> drp 3 ['a','b','c','d']
"d"
```

```elisp
(defun zip(l1 l2)
  (if (or (eq l1 nil) (eq l2 nil)) nil
    (let ((x (car l1)) (xs (cdr l1)) (y (car l2)) (ys (cdr l2)))
      (cons (cons x y) (zip xs ys)))))
(zip '(?a ?b ?c) '(1 2 3 4))
((97 . 1) (98 . 2) (99 . 3))

(defun drop(n l)
  (if (or (= n 0) (eq l nil)) l
    (drop (1- n) (cdr l))))
(concat (drop 3 '(?a ?b ?c ?d)))
"d"
```

```rust
>> fn drp<T>(n: i32, l: &[T]) -> &[T] { match (n, l) { (0, _) => l, (_, []) => &[], (n,[_, xs@..]) => drp(n-1, xs) } }
>> drp(3, &['a','b','c','d']).iter().collect::<String>()
"d"
```

## 6.4 Multiple recursion

`chapter6-functions.hs`
```haskell
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-2) + fib (n-1)

qsort :: Ord a => [a] -> [a]
qsort [] = []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
               where
                 smaller = [a | a <- xs, a <= x]
                 larger  = [b | b <- xs, b > x]
```

```haskell
λ> fib 10
55
λ> qsort [3,1,2,5,4]
[1,2,3,4,5]
```

```elisp
(defun fib(n)
  (cond ((= n 0) 0)
        ((= n 1) 1)
        (t (+ (fib (- n 2)) (fib (- n 1))))))
(fib 10)
55

(defun qsort(l)
  (if (eq l nil) l
    (let* ((x (car l))
           (xs (cdr l))
           (smaller (seq-filter (lambda (a) (<= a x)) xs))
           (larger (seq-filter (lambda (b) (> b x)) xs)))
      (append (qsort smaller) (list x) (qsort larger)))))
(qsort '(3 1 2 5 4))
(1 2 3 4 5)
```

```rust
>> fn fib(n: i32) -> i32 { if (n == 0 || n == 1) { n } else { fib(n-2) + fib(n-1) } }
>> fib(10)
55

>> fn qsort<T: Ord + Clone>(xs: &[T]) -> Vec<T> { if xs.is_empty() { vec![] } else { let x = &xs[0]; let (smaller, larger): (Vec<T>, Vec<T>) = xs[1..].iter().cloned().partition(|a| a <= x); let mut out = qsort(&smaller); out.push(x.clone()); out.extend(qsort(&larger)); out } }
>> qsort(&[3,1,2,5,4])
[1, 2, 3, 4, 5]
```

## 6.5 Mutual recursion

`chapter6-functions.hs`
```haskell
evn :: Int -> Bool
evn 0 = True
evn n = oddd(n-1)

oddd :: Int -> Bool
oddd 0 = False
oddd n = evn(n-1)

evens :: [a] -> [a]
evens []     = []
evens (x:xs) = x : odds xs

odds :: [a] -> [a]
odds [] = []
odds (_:xs) = evens xs
```

```haskell
λ> evn 4
True
λ> oddd 4
False
λ> evens "abcde"
"ace"
```

```elisp
(defun even(n)
  (if (= n 0) 't (odd (1- n))))
(defun odd(n)
  (if (= n 0) nil (even (1- n))))
(even 4)
t
(odd 4)
nil

(defun evens(l)
  (if (eq l nil) l (cons (car l) (odds (cdr l)))))
(defun odds(l)
  (if (eq l nil) l (evens (cdr l))))
(evens '(1 2 3 4 5))
(1 3 5)
```

```rust
>> fn odd(n: i32) -> bool { true }
>> fn even(n: i32) -> bool { match n { 0 => true, _ => odd(n-1) } }
>> fn odd(n: i32) -> bool { match n { 0 => false, _ => even(n-1) } }
>> even(4)
true
>> odd(4)
false
```

**Note:** In `evcxr` specifically, there was a need to "break" the recursive definition loop. In a real program this should not be needed.

```rust
>> fn odds(v: &[i32]) -> Vec<i32> { v.to_vec() }
>> fn evens(v: &[i32]) -> Vec<i32> { if v.is_empty() { vec![] } else { let mut x = v.to_vec(); let xs = x.split_off(1); x.extend(&odds(&xs)[..]); x } }
>> fn odds(v: &[i32]) -> Vec<i32> { if v.is_empty() { vec![] } else { let mut x = v.to_vec(); let xs = x.split_off(1); evens(&xs[..]) } }
>> evens(&[1,2,3,4,5])
[1, 3, 5]
```

## 6.8 Exercises

`chapter6-functions.hs`
```haskell
fac2 :: Int -> Int
fac2 0 = 1
fac2 n | n > 0 = n * fac2(n-1)
       | otherwise = 0

sumdown :: Int -> Int
sumdown 0 = 0
sumdown n = n + sumdown(n-1)

euclid :: Int -> Int -> Int
euclid m n | m == n = m
           | m > n  = euclid (m-n) n
           | m < n  = euclid m (n-m)

my_and :: [Bool] -> Bool
my_and []      = False
my_and [False] = False
my_and [True]  = True
my_and (x:xs)  = x && my_and xs

my_concat :: [[a]] -> [a]
my_concat [] = []
my_concat (l:ls) = l ++ my_concat ls

my_replicate :: Int -> a -> [a]
my_replicate 0 _ = []
my_replicate n x = x : my_replicate (n-1) x

my_nth :: [a] -> Int -> a
my_nth (x:_) 0 = x
my_nth (x:xs) n = my_nth xs (n-1)

my_elem :: Eq a => a -> [a] -> Bool
my_elem _ [] = False
my_elem x (y:ys) | x == y    = True
                 | otherwise = my_elem x ys

merge :: Ord a => [a] -> [a] -> [a]
merge l  [] = l
merge [] l  = l
merge (l:ls) (m:ms) | l <= m    = [l] ++ merge ls (m:ms)
                    | otherwise = [m] ++ merge (l:ls) ms

halve :: [a] -> ([a],[a])
halve l = (take n2 l, drop n2 l)
  where n2 = (length l) `div` 2

msort :: Ord a => [a] -> [a]
msort []  = []
msort [a] = [a]
msort l = merge (msort l1) (msort l2)
  where (l1, l2) = halve l
```

```haskell
λ> fac2(-1)
0
λ> fac2(-10)
0
λ> fac2 10
3628800
λ> sumdown 3
6
λ> euclid 6 27
3
λ> euclid 13 15
1
λ> euclid 13 27
1
λ> my_and [True,True,True]
True
λ> my_and [True,False,True]
False
λ> my_and []
False
λ> my_concat [[1,2],[3,4],[5]]
[1,2,3,4,5]
λ> my_concat [[1,2]]
[1,2]
λ> my_concat []
[]
λ> my_concat [[1]]
[1]
λ> my_replicate 5 'x'
"xxxxx"
λ> [1,2,3,4,5] `my_nth` 2
3
λ> my_elem 3 [1,2,3,4,5]
True
λ> my_elem 10 [1,2,3,4,5]
False
λ> merge [] [1,3,4]
[1,3,4]
λ> merge [2] [1,3,4]
[1,2,3,4]
λ> merge [2] [1]
[1,2]
λ> merge [3,4] [1,2,5]
[1,2,3,4,5]
λ> merge [2,5,6] [1,3,4]
[1,2,3,4,5,6]
λ> halve []
([],[])
λ> halve [1]
([],[1])
λ> halve [1,2]
([1],[2])
λ> halve [1,2,3]
([1],[2,3])
λ> msort []
[]
λ> msort [1]
[1]
λ> msort [2,1]
[1,2]
λ> msort [4,3,1,2,5]
[1,2,3,4,5]
```

```elisp
(defun fac2(n)
  (cond ((= n 0) 1)
        ((> n 0) (* n (fac (1- n))))
        ('t 0)))
(fac2 -1)
0
(fac2 -10)
0
(fac2 10)
3628800

(defun sumdown(n)
  (if (= n 0) n (+ n (sumdown (1- n)))))
(sumdown 3)
6

(defun euclid(m n)
  (cond ((= m n) m)
        ((> m n) (euclid (- m n) n))
        ((< m n) (euclid m (- n m)))))
(euclid 6 27)
3
(euclid 13 15)
1
(euclid 13 27)
1
(euclid 15 5)
5

(defun my-and(l)
  (pcase l
    ('nil nil)
    ('(nil) nil)
    ('(t) t)
    (`(,x . ,xs) (and x (my-and xs)))))
(my-and '(t t t))
t
(my-and '(t nil t))
nil
(my-and '())
nil

(defun replicate(n x)
  (if (= n 0) nil (cons x (replicate (1- n) x))))
(concat (replicate 5 ?x))
"xxxxx"

(defun my-nth(l n)
  (if (= n 0) (car l) (my-nth (cdr l) (1- n))))
(my-nth '(1 2 3 4 5) 2)
3

(defun elem(x l)
  (pcase l
    ('nil nil)
    (`(,y . ,ys) (if (equal x y) 't (elem x ys)))))
(elem 3 '(1 2 3 4 5))
t
(elem 10 '(1 2 3 4 5))
nil

(defun my-merge(x y)
  (cond ((eq x nil) y)
        ((eq y nil) x)
        ((<= (car x) (car y)) (cons (car x) (my-merge (cdr x) y)))
        ('t (cons (car y) (my-merge x (cdr y))))))
(my-merge '() '(1 3 4))
(1 3 4)
(my-merge '(2) '(1 3 4))
(1 2 3 4)
(my-merge '(2) '(1))
(1 2)
(my-merge '(3 4) '(1 2 5))
(1 2 3 4 5)
(my-merge '(2 5 6) '(1 3 4))
(1 2 3 4 5 6)

(defun halve(l)
  (let ((n2 (/ (length l) 2)))
    (list (take n2 l) (nthcdr n2 l))))
(halve nil)
(nil nil)
(halve '(1))
(nil (1))
(halve '(1 2))
((1) (2))
(halve '(1 2 3))
((1) (2 3))

(car '((1) (2 3)))
(1)
(cadr '((1) (2 3)))
(2 3)

(defun msort(l)
  (let ((ll (length l)))
    (if (<= ll 1) l
      (let ((hl (halve l)))
        (my-merge (msort (car hl)) (msort (cadr hl)))))))
(msort '())
nil
(msort '(1))
(1)
(msort '(2 1))
(1 2)
(msort '(4 3 1 2 5))
(1 2 3 4 5)
```

```rust
>> fn fac2(n: i32) -> i32 { match n { 0 => 1, 1.. => n * fac2(n-1), _ => 0 } }
>> fac2(-1)
0
>> fac2(-10)
0
>> fac2(10)
3628800

>> fn sumdown(n: i32) -> i32 { match n { 0 => 0, _ => n + sumdown(n-1) } }
>> sumdown(3)
6

>> fn euclid(m: i32, n: i32) -> i32 { if m == n { m } else if m > n { euclid(m-n, n) } else { euclid(m, n-m) } }
>> euclid(6, 27)
3
>> euclid(13, 15)
1
>> euclid(13, 27)
1

>> fn and(l: &[bool]) -> bool { match l { [] => false, [false] => false, [true] => true, [x, xs @ ..] => *x && and(xs) } }
>> and(&[true, true, true])
true
>> and(&[true, false, true])
false
>> and(&[])
false

>> fn concat<T: Clone>(lists: &[&[T]]) -> Vec<T> { if lists.is_empty() { vec![] } else { let mut result = lists[0].to_vec(); result.extend(concat(&lists[1..])); result } }
>> concat(&[&[1, 2], &[3, 4], &[5]])
[1, 2, 3, 4, 5]
>> concat(&[&[1,2]])
[1, 2]
>> concat(&[&[] as &[i32]])
[]
>> concat(&[&[1]])
[1]

>> fn replicate<T: Copy>(n: i32, a: T) -> Vec<T> { if n == 0 { vec![] } else { let mut r = vec![a]; r.extend(replicate(n-1, a).into_iter()); r } }
>> replicate(5, 2)
[2, 2, 2, 2, 2]

>> fn nth<T: Copy>(l: &[T], n: i32) -> Option<T> { if n == 0 { l.first().copied() } else { match l { [] => Option::default(), [x, xs @ ..] => nth(xs, n-1) } } }
>> nth(&[1,2,3,4,5], 2)
Some(3)
>> nth(&[1,2,3,4,5], 10)
None

>> fn elem<T: PartialEq>(x: &T, l: &[T]) -> bool { match l { [] => false, [y, ys @ ..] => if x == y { true } else { elem(x, ys) } } }
>> elem(&3, &[1,2,3,4,5])
true
>> elem(&10, &[1,2,3,4,5])
false

fn merge<T: std::cmp::PartialOrd + Clone>(l: &[T], m: &[T]) -> Vec<T> { if l.is_empty() { m.to_vec() } else if m.is_empty() { l.to_vec() } else if l[0] <= m[0] { let mut res = vec![l[0].clone()]; res.extend(merge(&l[1..], m)); res } else { let mut res = vec![m[0].clone()]; res.extend(merge(l, &m[1..])); res } }
>> merge(&[], &[1,3,4])
[1, 3, 4]
>> merge(&[2], &[1,3,4])
[1, 2, 3, 4]
>> merge(&[2], &[1])
[1, 2]
>> merge(&[3,4], &[1,2,5])
[1, 2, 3, 4, 5]
>> merge(&[2,5,6], &[1,3,4])
[1, 2, 3, 4, 5, 6]

>> fn halve<T: Clone>(l: &[T]) -> (Vec<T>,Vec<T>) { let n2 = l.len() / 2; let mut v = l.to_vec(); let v2 = v.split_off(n2); (v, v2) }
>> halve(&[] as &[i32])
([], [])
>> halve(&[1])
([], [1])
>> halve(&[1,2])
([1], [2])
>> halve(&[1,2,3])
([1], [2, 3])

>> fn msort<T: std::cmp::PartialOrd + Clone>(a: &[T]) -> Vec<T> { if a.is_empty() { vec![] } else if a.len() == 1 { a.to_vec() } else { let lr = halve(a); merge(&msort(&lr.0), &msort(&lr.1)) } }
>> msort(&[] as &[i32])
[]
>> msort(&[1])
[1]
>> msort(&[2,1])
[1, 2]
>> msort(&[4,3,1,2,5])
[1, 2, 3, 4, 5]
```
