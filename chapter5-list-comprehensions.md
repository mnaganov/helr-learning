# 5 List comprehensions

## 5.1 Basic concepts

```haskell
λ> [x^2 | x <- [1..5]]
[1,4,9,16,25]
λ> [(x,y) | x <- [1,2,3], y <- [4,5]]
[(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]
λ> [(x,y) | y <- [4,5], x <- [1,2,3]]
[(1,4),(2,4),(3,4),(1,5),(2,5),(3,5)]
λ> [(x,y) | x <- [1..3], y <- [x..3]]
[(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]
```

`chapter5-functions.hs`
```haskell
concat :: [[a]] -> [a]
concat xss = [x | xs <- xss, x <- xs]

firsts :: [(a,b)] -> [a]
firsts ps = [x | (x,_) <- ps]

length_sum :: [a] -> Int
length_sum xs = sum [1 | _ <- xs]
```

```haskell
λ> Main.concat [[1,2],[3,4],[5],[]]
[1,2,3,4,5]

λ> firsts [(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]
[1,1,2,2,3,3]

λ> length_sum [1,2,3,4,5]
5
λ> length_sum []
0
```

```elisp
(seq-map (lambda (x) (expt x 2)) (number-sequence 1 5))
(1 4 9 16 25)
```

Elisp lacks a built-in function for emitting a cartesian product. We can emulate
it by nested application of list iteration. However, another issue is that in
Lisp "efficient" addition of an element to the list (without copying the list)
is only possible to the head of the list—this is called "consing on the
list"—thus, the resulting list contans added elements in the reverse order.

Note that there is `generator.el` package intended to help with defining
generator functions. However, a generator function just emits value by value,
it does not store them anywhere. Thus, in order to generate a list of numbers
efficiently we need to do something like:

```elisp
(require 'generator)

(iter-defun my-iter-rev (from to)
  (dotimes (i (- (1+ to) from))
    (iter-yield (- to i))))
(let ((l nil))
  (iter-do (i (my-iter-rev 1 5)) (setq l (cons i l)))
  l)
(1 2 3 4 5)
```

The benefit of generators is that they can be "endless", however since Elisp
lacks "lazy evaluation", this does not help much. Concluding, since the code
using `generator` is not very convenient or readable, we will just use
`number-sequence` to emulate list comprehensions.

```elisp
(seq-reduce (lambda (r x)
              (seq-reduce (lambda (l e) (cons (list x e) l)) '(5 4) r))
            '(3 2 1) '())
((1 4) (1 5) (2 4) (2 5) (3 4) (3 5))

(seq-reduce (lambda (r x)
              (seq-reduce (lambda (l e) (cons (list e x) l)) '(3 2 1) r))
            '(5 4) '())
((1 4) (2 4) (3 4) (1 5) (2 5) (3 5))

(reverse
 (seq-reduce (lambda (r x)
               (seq-reduce (lambda (l e) (cons (list x e) l)) (number-sequence x 3) r))
             (number-sequence 1 3) '()))
((1 1) (1 2) (1 3) (2 2) (2 3) (3 3))

(defun lists-concat(xss)
  (nreverse
   (seq-reduce (lambda (r x)
                 (seq-reduce (lambda (l e) (cons e l)) x r))
               xss '())))
(lists-concat '((1 2) (3 4) (5) ()))
(1 2 3 4 5)

(defun firsts(ps)
  (seq-map (lambda (e) (seq-let (x _) e x)) ps))
(1 1 2 2 3 3)
```

**Note 1:** The macro `seq-let` is used for destructuring binding, similar to
`pcase`.

**Note 2:** When the length of the output list is the same as the length of the
input list, it is more practical to use `seq-map` which builds the result by
applying the provided function to each element of the input sequence.

```elisp
(require 'dash)
(defun map-length(xs)
  (-sum (seq-map (lambda (x) 1) xs)))
(map-length '(1 2 3 4 5))
5
(map-length '())
0
```

```rust
>> (1i32..=5).map(|x| x.pow(2)).collect::<Vec<_>>()
[1, 4, 9, 16, 25]
```

For cartesian products, we need to use crate `itertools`. In `evcxr`, this crate
must first be declared as an extern, and then imported into the namespace:

```rust
>> extern crate itertools;
   Compiling either v1.13.0
   Compiling itertools v0.14.0
>> use itertools::Itertools;
```

Then we can use it for iterators:

```rust
>> [1,2,3].iter().cartesian_product([4,5]).collect::<Vec<_>>()
[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
>> [4,5].iter().cartesian_product([1,2,3]).map(|(x,y)| (y,x)).collect::<Vec<_>>()
[(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)]
>> (1..=3).map(move |x| (x..=3).map(move |y| (x,y))).flatten().collect::<Vec<_>>()
[(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
```

after writing the last comprehension, I have realized that we could use the
same approach for generating the cartesian product:

```rust
>> [1,2,3].iter().map(move |x| [4,5].iter().map(move |y| (x,y))).flatten().collect::<Vec<_>>()
[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
```

In fact, `flatten` implements the `concat` function we had to write.

```rust
>> fn firsts(ps: &[(i32,i32)]) -> Vec<i32> { ps.into_iter().map(|(x,y)| *x).collect() }
>> firsts(&[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)])
[1, 1, 2, 2, 3, 3]

>> fn map_length(xs: &[i32]) -> usize { xs.into_iter().map(|_| 1usize).sum() }
>> map_length(&[1,2,3,4,5])
5
>> map_length(&[])
0
```

## 5.2 Guards

`chapter5-functions.hs`
```haskell
factors :: Int -> [Int]
factors n = [x | x <- [1..n], n `mod` x == 0]

prime :: Int -> Bool
prime n = factors n == [1,n]

primes :: Int -> [Int]
primes n = [x | x <- [2..n], prime x]

find :: Eq a => a -> [(a,b)] -> [b]
find k t = [v | (k',v) <- t, k == k']
```

```haskell
λ> factors 15
[1,3,5,15]
λ> factors 7
[1,7]
λ> prime 15
False
λ> prime 7
True
λ> primes 40
[2,3,5,7,11,13,17,19,23,29,31,37]

λ> find 'b' [('a',1),('b',2),('c',3),('b',4)]
[2,4]
```

```elisp
(defun factors(n)
  (seq-filter (lambda (x) (= (% n x) 0)) (number-sequence 1 n)))
(factors 15)
(1 3 5 15)
(factors 7)
(1 7)

(defun prime(n) (equal (factors n) (list 1 n)))
(prime 15)
nil
(prime 7)
t

(defun primes(n) (seq-filter #'prime (number-sequence 2 n)))
(primes 40)
(2 3 5 7 11 13 17 19 23 29 31 37)

(defun find-all-in-map(k t)
  (nreverse
   (seq-reduce (lambda (l p) (if (equal (car p) k) (cons (cdr p) l) l)) t '())))
(find-all-in-map ?b '((?a . 1) (?b . 2) (?c . 3) (?b . 4)))
(2 4)
```

**Note:** Although Elisp lacks the "tuple" type, pairs can be represented as
"cons cells", and they are often used for implementing property lists.

```rust
>> fn factors(n: i32) -> Vec<i32> { (1..=n).filter(|x| n % x == 0).collect() }
>> factors(15)
[1, 3, 5, 15]
>> factors(7)
[1, 7]

>> fn prime(n: i32) -> bool { factors(n) == [1,n] }
>> prime(15)
false
>> prime(7)
true

>> fn primes(n: i32) -> Vec<i32> { (2..=n).filter(|x| prime(*x)).collect() }
>> primes(40)
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
```

**Note:** In order to be able to pass `prime` directly to `filter`, it must
take a reference. The same functions could be defined as follows:

```rust
>> fn factors(n: &i32) -> Vec<i32> { let m = *n; (1..=m).filter(|x| m % x == 0).collect() }
>> factors(&15)
[1, 3, 5, 15]
>> fn prime(n: &i32) -> bool { factors(n) == [1,*n] }
>> prime(&15)
false
>> fn primes(n: &i32) -> Vec<i32> { let m=*n; (2..=m).filter(prime).collect() }
>> primes(&40)
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
>> fn find_all_in_map(k: char, t: &[(char, i32)]) -> Vec<i32> { let (els, vals): (Vec<_>, Vec<_>) = t.iter().filter(|(e,v)| k == *e).cloned().unzip(); vals }
>> find_all_in_map('b', &[('a', 1),('b',2),('c',3),('b',4)])
[2, 4]
```

## 5.3 The `zip` function

```haskell
λ> zip ['a','b','c'] [1,2,3,4]
[('a',1),('b',2),('c',3)]
```

`chapter5-functions.hs`
```haskell
pairs :: [a] -> [(a,a)]
pairs xs = zip xs (tail xs)

sorted :: Ord a => [a] -> Bool
sorted xs = and [x <= y | (x,y) <- pairs xs]

positions :: Eq a => a -> [a] -> [Int]
positions x xs = [i | (x',i) <- zip xs [0..], x == x']
```

```haskell
λ> pairs [1,2,3,4]
[(1,2),(2,3),(3,4)]
λ> sorted [1,2,3,4]
True
λ> sorted [1,3,2,4]
False
λ> positions False [True, False, True, False]
[1,3]
```

```elisp
(mapcar* #'cons '(?a ?b ?c) '(1 2 3 4))
((97 . 1) (98 . 2) (99 . 3))

(defun pairs(xs)
  (mapcar* #'cons xs (cdr xs)))
(pairs '(1 2 3 4))
((1 . 2) (2 . 3) (3 . 4))

(defun sorted/pairs(xs)
  (seq-every-p (lambda (c) (<= (car c) (cdr c))) (pairs xs)))
(sorted/pairs '(1 2 3 4))
t
(sorted/pairs '(1 3 2 4))
nil
```

**Note:** Since Elisp does not support lazy evaluation like Haskell, simply
copying the Haskell approach to implementation produces an inefficient
solution. Also, since Elisp also does not support tail call optimization (see
"Evolution of Emacs Lisp"), we can not write an efficient recursive version
either. Below is a less verbose but more efficient imperative implementation
which exits early when if finds the first out of order pair:

```elisp
(defun sorted/loop(xs)
  (let ((sorted 't))
    (unless (eq xs nil)
      (let ((e (car xs)) (l (cdr xs)))
        (while (and sorted (not (eq l nil)))
          (if (> e (car l)) (setq sorted nil)
            (progn
              (setq e (car l))
              (setq l (cdr l)))))))
    sorted))
(sorted/loop '())
t
(sorted/loop '(1))
t
(sorted/loop '(1 2 3 4))
t
(sorted/loop '(1 3 2 4))
nil

(defun positions(x xs)
  (nreverse (let ((i -1))
    (seq-reduce (lambda (l e)
                  (setq i (+ i 1))
                  (if (equal e x) (cons i l) l)) xs nil))))
(positions nil '(t nil t nil))
(1 3)

(defun positions/indexed(x xs)
  (seq-map 'cadr
       (seq-filter (lambda (p) (equal x (car p)))
                   (seq-map-indexed (lambda (e idx) (list e idx)) xs))))
```

**Note:** The second version (`positions/indexed`) is closer in the spirit
to its Haskell prototype.

```rust
>> ['a','b','c'].iter().zip([1,2,3,4].iter()).collect::<Vec<_>>()
[('a', 1), ('b', 2), ('c', 3)]

>> fn pairs(xs: &[i32]) -> Vec<(&i32,&i32)> { let mut xs1 = xs.iter(); xs1.next(); xs.iter().zip(xs1).collect::<Vec<_>>() }
>> pairs(&[1,2,3,4])
[(1, 2), (2, 3), (3, 4)]

>> fn sorted(xs: &[i32]) -> bool { pairs(xs).iter().all(|&(x,y)| x<=y) }
>> sorted(&[1,2,3,4])
true
>> sorted(&[1,3,2,4])
false
```

**Note:** Once again, this is not a lazy implementation because `pairs` produces
all the pairs. There is `is_sorted_by` method which implements iterative
comparison, and `is_sorted` which uses standard `<=` comparator.

```rust
>> [1,2,3,4].iter().is_sorted()
true
>> [1,3,2,4].iter().is_sorted()
false

>> fn positions(x: i32, xs: &[i32]) -> Vec<usize> { let (idxs, elts): (Vec<_>, Vec<_>) = (0_usize..).zip(xs.iter()).filter(|(i,e)| *e == &x).unzip::<usize, &i32, Vec<usize>, Vec<&i32>>(); idxs }
>> positions(2, &[1,2,1,2])
[1, 3]
```

**Note:** "Zipping" onto open intervals works, but not vice versa. There is a
standard method `enumerate` which does a similar thing.

```rust
>> fn positions_enum(x: i32, xs: &[i32]) -> Vec<usize> { let (idxs, elts): (Vec<_>, Vec<_>) = xs.iter().enumerate().filter(|(i,e)| *e == &x).unzip::<usize, &i32, Vec<usize>, Vec<&i32>>(); idxs }
>> positions_enum(2, &[1,2,1,2])
[1, 3]
```

## 5.4 String comprehensions

`chapter5-functions.hs`
```haskell
lowers :: String -> Int
lowers xs = length [x | x <- xs, x >= 'a' && x <= 'z']

count :: Char -> String -> Int
count x xs = length [x' | x' <- xs, x == x']
```

```haskell
λ> "abcde" !! 2
'c'
λ> take 3 "abcde"
"abc"
λ> length "abcde"
5
λ> zip "abc" [1,2,3,4]
[('a',1),('b',2),('c',3)]
λ> lowers "Haskell"
6
λ> count 's' "Mississippi"
4
```

```elisp
(elt "abcde" 2)
99
(string (elt "abcde" 2))
"c"
(seq-take "abcde" 3)
"abc"
(length "abcde")
5
(mapcar* #'cons "abc" '(1 2 3 4))
((97 . 1) (98 . 2) (99 . 3))
(defun lowers(xs)
  (length (seq-filter (lambda (e) (and (>= e ?a) (<= e ?z))) xs)))
(lowers "Haskell")
6
(defun count_char(x xs)
  (length (seq-filter (lambda (e) (equal e x)) xs)))
(count_char ?s "Mississippi")
4
```

Rust does not support string indexing in the "classic" way. That's because
strings use UTF-8 where each Unicode character can take arbitrary number of
bytes. This makes the complexity of the character indexing function to be
linear. So instead, Rust forces the user to choose the desired string access
mode by calling one of `as_...` functions which returns an iterator.

```rust
>> "abcde".chars().nth(2)
Some('c')
>> "abcde".chars().take(3).collect::<String>()
"abc"
>> "abcde".chars().count()
5
>> "abc".chars().zip((1..=4)).collect::<Vec<_>>()
[('a', 1), ('b', 2), ('c', 3)]
>> fn lowers(xs: &str) -> usize { xs.chars().filter(|x| x >= &'a' && x <= &'z').count() }
>> lowers("Haskell")
6
>> fn count(x: &char, xs: &str) -> usize { xs.chars().filter(|c| c == x).count() }
>> count(&'s', "Mississippi")
4
```

## 5.5 The Caesar cipher

`chapter5-functions.hs`
```haskell
import Data.Char

let2int :: Char -> Int
let2int c = ord c - ord 'a'

int2let :: Int -> Char
int2let n = chr (ord 'a' + n)

shift :: Int -> Char -> Char
shift n c | isLower c = int2let ((let2int c + n) `mod` 26)
          | otherwise = c

encode :: Int -> String -> String
encode n xs = [shift n x | x <- xs]

table :: [Float]
table = [8.1, 1.5, 2.8, 4.2, 12.7, 2.2, 2.0, 6.1, 7.0,
         0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0,
         6.3, 9.0, 2.8, 1.0, 2.4, 0.2, 2.0, 0.1]

percent :: Int -> Int -> Float
percent n m = (fromIntegral n / fromIntegral m) * 100

freqs :: String -> [Float]
freqs xs = [percent (count x xs) n | x <- ['a'..'z']] where n = lowers xs

chisqr :: [Float] -> [Float] -> Float
chisqr os es = sum [((o-e)^2)/e | (o,e) <- zip os es]

rotate :: Int -> [a] -> [a]
rotate n xs = drop n xs ++ take n xs

crack :: String -> String
crack xs = encode (-factor) xs
  where
    factor = head (positions (minimum chitab) chitab)
    chitab = [chisqr (rotate n table') table | n <- [0..25]]
    table' = freqs xs
```

```haskell
λ> let2int 'a'
0
λ> int2let 0
'a'
λ> shift 3 'a'
'd'
λ> shift 3 'z'
'c'
λ> shift (-3) 'c'
'z'
λ> shift 3 ' '
' '
λ> encode 3 "haskell if fun"
"kdvnhoo li ixq"
λ> encode (-3) "kdvnhoo li ixq"
"haskell if fun"
λ> percent 5 15
33.333336
λ> freqs "abbcccddddeeeee"
[6.666667,13.333334,20.0,26.666668,33.333336,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
λ> chisqr [1.0, 2.0, 3.0] [3.0, 2.0, 1.0]
5.3333335
λ> chisqr [1.0, 2.0, 3.0] [2.0, 3.0, 1.0]
4.8333335
λ> rotate 3 [1,2,3,4,5]
[4,5,1,2,3]
λ> crack "kdvnhoo lv ixq"
"haskell is fun"
λ> crack "vscd mywzboroxcsyxc kbo ecopev"
"list comprehensions are useful"
λ> crack (encode 3 "haskell")
"piasmtt"
λ> crack (encode 3 "boxing wizards jump quickly")
"wjsdib rduvmyn ephk lpdxfgt"
```

```elisp
(defun let2int(c) (- c ?a))
(let2int ?a)
0
(defun int2let(n) (+ ?a n))
(int2let 0)
97
(defun isLower(c) (and (>= c ?a) (<= c ?z)))
(isLower ?c)
t
(defun shift(n c)
  (if (isLower c) (int2let (mod (+ (let2int c) n) 26)) c))
(string (shift 3 ?a))
"d"
(string (shift 3 ?z))
"c"
(string (shift -3 ?c))
"z"
(string (shift 3 ? ))
" "

(defun encode(n xs)
  (concat (mapcar (lambda (x) (shift n x)) xs)))
(encode 3 "haskell if fun")
"kdvnhoo li ixq"
(encode -3 "kdvnhoo li ixq")
"haskell if fun"

(setq table '(8.1 1.5 2.8 4.2 12.7 2.2 2.0 6.1 7.0
              0.2 0.8 4.0 2.4 6.7 7.5 1.9 0.1 6.0
              6.3 9.0 2.8 1.0 2.4 0.2 2.0 0.1))
(8.1 1.5 2.8 4.2 12.7 2.2 2.0 6.1 7.0 0.2 0.8 4.0 ...)

(defun percent(n m) (* (/ (float n) (float m)) 100))
(percent 5 15)
33.33333333333333

(defun freqs(xs)
  (let ((n (lowers xs)))
    (mapcar (lambda (x) (percent (count x xs) n)) (number-sequence ?a ?z))))
(freqs "abbcccddddeeeee")
(6.666666666666667 13.333333333333334 20.0 26.666666666666668 33.33333333333333 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...)

(require 'dash)
(defun chisqr(os es)
  (-sum (seq-mapn (lambda (o e) (/ (expt (- o e) 2) e)) os es)))
(chisqr '(1.0 2.0 3.0) '(3.0 2.0 1.0))
5.333333333333333
(chisqr '(1.0 2.0 3.0) '(2.0 3.0 1.0))
4.833333333333333

(defun rotate(n xs)
  (append (nthcdr n xs) (take n xs)))
(rotate 3 '(1 2 3 4 5))
(4 5 1 2 3)

(defun crack(xs)
  (let* ((ftable (freqs xs))
         (chitab (mapcar (lambda (n) (chisqr (rotate n ftable) table))
                         (number-sequence 0 25)))
         (factor (car (positions (seq-min chitab) chitab))))
    (encode (- factor) xs)))
(crack "kdvnhoo lv ixq")
"haskell is fun"
(crack "vscd mywzboroxcsyxc kbo ecopev")
"list comprehensions are useful"
(crack (encode 3 "haskell"))
"piasmtt"
(crack (encode 3 "boxing wizards jump quickly"))
"wjsdib rduvmyn ephk lpdxfgt"
```

```rust
>> fn let2int(c: char) -> i32 { (c as i32) - ('a' as i32) }
>> let2int('a')
>> fn int2let(n: i32) -> char { match char::from_u32(('a' as i32 + n) as u32) { None => '?', Some(c) => c } }
>> int2let(0)
'a'

>> fn shift(n: i32, c: char) -> char { if c.is_lowercase() { int2let((let2int(c) + n).rem_euclid(26)) } else { c } }
>> shift(3, 'a')
'd'
>> shift(3, 'z')
'c'
>> shift(-3, 'c')
'z'
>> shift(-3, ' ')
' '

>> fn encode(n: i32, xs: &str) -> String { xs.chars().map(|x| shift(n, x)).collect() }
>> encode(3, "haskell if fun")
"kdvnhoo li ixq"
>> encode(-3, "kdvnhoo li ixq")
"haskell if fun"

>> let table = [8.1, 1.5, 2.8, 4.2, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.0, 2.8, 1.0, 2.4, 0.2, 2.0, 0.1];

>> fn percent(n: i32, m: i32) -> f64 { (n as f64 / m as f64) * 100.0 }
>> percent(5, 15)
33.33333333333333

>> fn freqs(xs: &str) -> Vec<f64> { let n = lowers(xs); ('a'..='z').map(|x| percent(count(&x,xs) as i32, n as i32)).collect() }
>> freqs("abbcccddddeeeee")
[6.666666666666667, 13.333333333333334, 20.0, 26.666666666666668, 33.33333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

>> fn chisqr(os: &[f64], es: &[f64]) -> f64 { os.iter().zip(es).map(|(o, e)| ((o-e).powf(2.0))/e).sum() }
>> chisqr(&[1.0, 2.0, 3.0],&[3.0, 2.0, 1.0])
5.333333333333333
>> chisqr(&[1.0, 2.0, 3.0],&[2.0, 3.0, 1.0])
4.833333333333333

>> fn crack(xs: &str, table: &[f64]) -> String { let mut ftable = freqs(xs); let mut chitab: Vec<f64> = Vec::with_capacity(26); for n in 0..=25 { chitab.push(chisqr(&ftable, table)); ftable.rotate_left(1); }; let minchi = chitab.iter().fold(f64::INFINITY, |a, &b| a.min(b)); let factor = chitab.iter().position(|&x| x==minchi).unwrap(); encode(-(factor as i32), xs) }
>> crack("kdvnhoo lv ixq", &table)
"haskell is fun"
>> crack("vscd mywzboroxcsyxc kbo ecopev", &table)
"list comprehensions are useful"
>> crack(&encode(3, "haskell"), &table)
"piasmtt"
>> crack(&encode(3, "boxing wizards jump quickly"), &table)
"wjsdib rduvmyn ephk lpdxfgt"
```

**Note:** We need to pass `table` to `crack` because for some reason in `evcxr`
functions do not have access to the global context.

## 5.7 Exercises

`chapter5-functions.hs`
```haskell
grid :: Int -> Int -> [(Int,Int)]
grid m n = [(x,y) | x <- [0..m], y <- [0..n]]

square :: Int -> [(Int,Int)]
square n = [(x,y) | (x,y) <- grid n n, x /= y]

replicate :: Int -> a -> [a]
replicate n x = [x | _ <- [1..n]]

pyths :: Int -> [(Int,Int,Int)]
pyths n = [(x,y,z) | x <- [1..n], y <- [1..n], z <- [1..n], x^2 + y^2 == z^2]

perfects :: Int -> [Int]
perfects n = [m | m <- [1..n], sum (factors m) == 2*m]

positions_find :: Eq a => a -> [a] -> [Int]
positions_find x xs = find x (zip xs [0..])

scalarproduct :: [Int] -> [Int] -> Int
scalarproduct xs ys = sum [x*y | (x,y) <- zip xs ys]
```

```haskell
λ> sum [x^2 | x <- [1..100]]
338350

λ> grid 1 2
[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

λ> square 2
[(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]

λ> Main.replicate 3 True
[True,True,True]

λ> pyths 10
[(3,4,5),(4,3,5),(6,8,10),(8,6,10)]

λ> perfects 500
[6,28,496]

λ> Main.concat [[(x,y)|y<-[3,4]]|x<-[1,2]]
[(1,3),(1,4),(2,3),(2,4)]

λ> positions_find False [True, False, True, False]
[1,3]

λ> scalarproduct [1,2,3] [4,5,6]
32
```

```elisp
(-sum (seq-map (lambda (x) (* x x)) (number-sequence 1 100)))
338350

(defun grid(m n)
  (seq-reduce (lambda (r x)
                (seq-reduce (lambda (l e) (cons (cons x e) l)) (number-sequence n 0 -1) r))
              (number-sequence m 0 -1) '()))
(grid 1 2)
((0 . 0) (0 . 1) (0 . 2) (1 . 0) (1 . 1) (1 . 2))

(defun square(n)
  (seq-filter (lambda (c) (not (equal (car c) (cdr c)))) (grid n n)))
(square 2)
((0 . 1) (0 . 2) (1 . 0) (1 . 2) (2 . 0) (2 . 1))

(defun replicate(n x)
  (seq-map (lambda (_) x) (number-sequence 1 n)))
(replicate 3 't)
(t t t)

(defun pyths(n)
  (let ((result nil)
        (x n)
        (y n)
        (z n))
    (while (>= x 1)
      (setq y n)
      (while (>= y 1)
        (setq z n)
        (while (>= z 1)
          (when (= (* z z) (+ (* x x) (* y y)))
            (setq result (cons (list x y z) result)))
          (setq z (1- z)))
        (setq y (1- y)))
      (setq x (1- x)))
    result))
(pyths 10)
((3 4 5) (4 3 5) (6 8 10) (8 6 10))

(defun perfects(n)
  (seq-filter (lambda (m) (= (-sum (factors m)) (* 2 m))) (number-sequence 1 n)))
(perfects 500)
(6 28 496)

(lists-concat
 (seq-map (lambda (x)
            (seq-map (lambda (y) (cons x y)) (number-sequence 3 4)))
          (number-sequence 1 2)))
((1 . 3) (1 . 4) (2 . 3) (2 . 4))

(defun positions/find(x xs)
  (find-all-in-map x (mapcar* #'cons xs (number-sequence 1 (length xs)))))
(positions/find nil '(t nil t nil))
(2 4)

(defun scalarproduct(xs ys)
  (-sum (mapcar* #'* xs ys)))
(scalarproduct '(1 2 3) '(4 5 6))
32
```

```rust
>> (1i32..=100).map(|x| x.pow(2)).sum::<i32>()
338350

>> fn grid(m: i32, n: i32) -> Vec<(i32,i32)> { (0..=m).map(move |x| (0..=n).map(move |y| (x,y))).flatten().collect::<Vec<(i32,i32)>>() }
>> grid(1,2)
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

>> fn square(n: i32) -> Vec<(i32,i32)> { grid(n, n).into_iter().filter(|(x,y)| x != y).collect::<Vec<(i32,i32)>>() }
>> square(2)
[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

>> fn replicate<T:Copy>(n: i32, x: T) -> Vec<T> { (1..=n).map(|_| x).collect::<Vec<T>>() }
>> replicate(3, true)
[true, true, true]

>> fn pyths(n: i32) -> Vec<(i32,i32,i32)> { (1..=n).map(move |x| (1..=n).map(move |y| (x,y))).flatten().map(|(x,y)| (1..=n).map(move |z| (x,y,z))).flatten().filter(|(x,y,z)| x*x+y*y==z*z).collect::<Vec<(i32,i32,i32)>>() }
>> pyths(10)
[(3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10)]
```

**Note:** `flatten` can only remove one level of indirection, thus it
is impossible to nest `3` generators.

```rust
>> fn perfects(n: i32) -> Vec<i32> { (1..=n).filter(|m| factors(m).iter().sum::<i32>() == 2*m).collect::<Vec<i32>>() }
>> perfects(500)
[6, 28, 496]

>> fn positions_find<T: PartialEq>(x: &T, xs: &[T]) -> Vec<usize> { xs.iter().zip(0..).filter_map(|(e, i)| if e == x { Some(i) } else { None }).collect() }
>> positions_find(&false, &[true, false, true, false])
[1, 3]

>> fn scalarproduct(xs: &[i32], ys: &[i32]) -> i32 { xs.iter().zip(ys.iter()).map(|(x,y)| x*y).sum() }
>> scalarproduct(&[1,2,3], &[4,5,6])
32
```
