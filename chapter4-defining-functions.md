# 4. Defining functions

## 4.1 New from old

`chapter4-functions.hs`
```haskell
even :: Integral a => a -> Bool
even n = n `mod` 2 == 0

splitAt :: Int -> [a] -> ([a],[a])
splitAt n xs = (take n xs, drop n xs)

recip :: Fractional a => a -> a
recip n = 1/n
```

```haskell
λ> Main.even 5
False
λ> Main.even 4
True
λ> Main.splitAt 3 [1, 2, 3, 4, 5]
([1,2,3],[4,5])
λ> Main.recip 2.0
0.5
λ> Main.recip 2
0.5
```

**Note:** since we define functions with the same name as in Prelude,
we need to qualify them with a prefix `Main.`.

```elisp
(defun even (n) (= (mod n 2) 0))
(even 5)
nil
(even 4)
t

(defun splitAt (n xs)
  (list (take n xs) (nthcdr n xs)))
(splitAt 3 '(1 2 3 4 5))
((1 2 3) (4 5))

(defun recip (n) (/ n))
(recip 2.0)
0.5
(recip 2)
0
```

```rust
>> fn even(n: u32) -> bool { n % 2 == 0 }
>> even(5)
false
>> even(4)
true

>> fn splitAt(n: usize, xs: &[i32]) -> (&[i32], &[i32]) { (&xs[0..n],&xs[n..]) }
>> splitAt(3, &[1,2,3,4,5])
([1, 2, 3], [4, 5])

>> fn recip(n: f64) -> f64 { 1.0_f64 / n }
>> recip(2.0)
0.5
```

## 4.2 Conditional expressions

`chapter4-functions.hs`
```haskell
abs_if :: Int -> Int
abs_if n = if n >= 0 then n else -n

signum_if :: Int -> Int
signum_if n = if n < 0 then -1 else
                if n == 0 then 0 else 1
```

```haskell
λ> abs_if (-5)
5
λ> abs_if 3
3
λ> signum_if (-5)
-1
λ> signum_if 3
1
```

**Note:** The function names have `_if` suffix because there will be
implementations which use guards.

```elisp
(defun abs/if (n) (if (>= n 0) n (- n)))
(abs/if -5)
5
(abs/if 3)
3

(defun signum/if (n)
  (if (< n 0) -1
    (if (= n 0) 0 1)))
(signum/if -5)
-1
(signum/if 3)
1
```

**Note 1:** In Lisp tradition, an alternative implementation uses `/` as a
delimiter for the variant name.

**Note 2:** To negate a variable, need to write `(- n)` because `-n` is
interpreted as a variable name.

```rust
>> fn abs_if(n: i32) -> i32 { if n >= 0 { n } else { -n } }
>> abs_if(-5)
5
>> abs_if(3)
3

>> fn signum_if(n: i32) -> i32 { if n < 0 { -1 } else if n == 0 { 0 } else { -1 } }
>> signum_if(-5)
-1
>> signum_if(3)
-1
```

## 4.3 Guarded equations

`chapter4-functions.hs`
```haskell
abs_grd :: Int -> Int
abs_grd n | n >= 0    = n
          | otherwise = -n

signum_grd :: Int -> Int
signum_grd n | n < 0     = -1
             | n == 0    = 0
             | otherwise = 1
```

```haskell
λ> abs_grd (-5)
5
λ> abs_grd 3
3
λ> signum_grd (-5)
-1
λ> signum_grd 3
1
```

```elisp
(defun abs/cond (n)
  (cond ((>= n 0) n)
        (t (- n))))
(abs/cond -5)
5
(abs/cond 3)
3

(defun signum/cond (n)
  (cond ((< n 0) -1)
        ((= n 0) 0)
        (t 1)))
(signum/cond -5)
-1
(signum/cond 3)
1
```

```rust
>> fn abs_match(n: i32) -> i32 { match n { 0.. => n, _ => -n } }
>> abs_match(-5)
5
>> abs_match(3)
3
>> fn signum_match(n: i32) -> i32 { match n { ..0 => -1, 0 => 0, _ => 1 } }
>> signum_match(-5)
-1
>> signum_match(3)
1
>> signum_match(0)
0
```

## 4.4 Pattern matching

`chapter4-functions.hs`
```haskell
not :: Bool -> Bool
not False = True
not True = False

and :: Bool -> Bool -> Bool
True `and` b = b
False `and` _ = False
```

```haskell
λ> Main.not False
True
λ> True `Main.and` True
True
λ> False `Main.and` True
False
```

```elisp
(defun not/pcase (x)
  (pcase x
    ('nil t)
    ('t nil)))

(not/pcase nil)
t
(not/pcase t)
nil
(not/pcase 5)
nil

(defun not/pcase_check (x)
  (pcase x
    ('nil t)
    ('t nil)
    (_ error "Invalid argument")))
```

**Note:** The second version only accepts boolean parameters, matching the
behavior of the Haskell version.

```elisp
(defun and/pcase (x y)
  (pcase (list x y)
    (`(t ,b) b)
    (`(nil ,any) 'nil)))
(and/pcase t t)
t
(and/pcase t nil)
nil
(and/pcase nil nil)
nil
(and/pcase nil t)
nil
```

```rust
>> fn not_match(b: bool) -> bool { match b { false => true, true => false } }
>> not_match(false)
true
>> not_match(true)
false
>> fn and_match(x: bool, y: bool) -> bool { match (x, y) { (true, b) => b, (false, _) => false } }
>> and_match(true, true)
true
>> and_match(true, false)
false
>> and_match(false, false)
false
>> and_match(false, true)
false
```

### Tuple patterns

`chapter4-functions.hs`
```haskell
fst :: (a,b) -> a
fst (x,_) = x

snd :: (a,b) -> b
snd (_,y) = y
```

```haskell
λ> Main.fst (1,2)
1
λ> Main.snd (1,2)
2
```

```elisp
(defun fst (xs)
  (pcase xs
    (`(,x ,rest) x)))
(defun snd (xs)
  (pcase xs
    (`(,head ,y) y)))

(fst '(1 2))
1
(snd '(1 2))
2
```

```rust
>> fn fst(p: (i32, i32)) -> i32 { match p { (x,_) => x } }
>> fst((1,2))
1
>> fn snd(p: (i32, i32)) -> i32 { match p { (_,y) => y } }
>> snd((1,2))
2
```

### List patterns

`chapter4-functions.hs`
```haskell
head :: [a] -> a
head (x:_) = x

tail :: [a] -> [a]
tail (_:xs) = xs
```

```haskell
λ> Main.head [1,2,3]
1
λ> Main.tail [1,2,3]
[2,3]
λ> Main.tail [1,2]
[2]
λ> Main.tail [1]
[]
```

```elisp
(defun head/pcase (lst)
  (pcase lst
    (`(,x . ,rest) x)))
(head/pcase '(1 2 3))
1
(head/pcase '())
nil

(defun tail/pcase (lst)
  (pcase lst
    (`(,h . ,xs) xs)))
(tail/pcase '(1 2 3))
(2 3)
(tail/pcase '(1 2))
(2)
(tail/pcase '(1))
nil
```

```rust
>> fn head_match<T>(l: &[T]) -> &T { match l { [x, ..] => x, [] => panic!() } }
>> head_match(&[1, 2, 3])
1
>> fn tail_match<T>(l: &[T]) -> &[T] { match l { [_, xs @ ..] => xs, [] => &[] } }
>> tail_match(&[1,2,3])
[2, 3]
>> tail_match(&[1,2])
[2]
>> tail_match(&[1])
[]
```

**Note:** Rust requires defining the arm for handling an empty list in the
`head` function.

## 4.5 Lambda expressions

`chapter4-functions.hs`
```haskell
add_l :: Int -> (Int -> Int)
add_l = \x -> (\y -> x + y)

const_l :: a -> (b -> a)
const_l x = \_ -> x

odds_l :: Int -> [Int]
odds_l n = map (\x -> x*2 + 1) [0..n-1]
```

```haskell
λ> add_l 2 3
5
λ> const_l 2 3
2
λ> odds_l 5
[1,3,5,7,9]
```

```elisp
(defun add_l (x)
  (lambda (y) (+ x y)))
(funcall (add_l 2) 3)
5
(apply (add_l 2) '(3))
5

(defun const_l (x)
  (lambda (_) x))
(funcall (const_l 2) 3)
2

(defun odds_l (n)
  (seq-map (lambda (x) (+ (* x 2) 1)) (number-sequence 0 (- n 1))))
(odds_l 5)
(1 3 5 7 9)
```

```rust
>> { let add_l = move |x| {move |y| x+y}; add_l(2)(3) }
5
>> { let const_l = move |x| {move |_| x}; const_l(2)(3) }
2
>> fn odds_l(n: i32) -> Vec<i32> { (0..n).map(|x| {x*2+1}).collect() }
>> odds_l(5)
[1, 3, 5, 7, 9]
```

## 4.6 Operator sections

`chapter4-functions.hs`
```haskell
sum :: [Int] -> Int
sum = foldl (+) 0
```

```haskell
λ> Main.sum [1,2,3,4,5]
15
```

```elisp
(defun sum (l)
  (seq-reduce #'+ l 0))
(sum '(1 2 3 4 5))
15
```

```rust
>> fn sum(l: &[i32]) -> i32 { l.iter().fold(0, |acc, x| acc + x) }
>> sum(&[1,2,3,4,5])
15
```

## 4.8 Exercises

`chapter4-functions.hs`
```haskell
halve :: [a] -> ([a],[a])
halve xs = Main.splitAt (length xs `div` 2) xs

third_a :: [a] -> a
third_a xs = Main.head (Main.tail (Main.tail xs))

third_b :: [a] -> a
third_b xs = xs !! 2

third_c :: [a] -> a
third_c (_:(_:(x:_))) = x

safetail_a :: [a] -> [a]
safetail_a xs = if null xs then xs else Main.tail xs

safetail_b :: [a] -> [a]
safetail_b xs | null xs = xs
              | otherwise = Main.tail xs

safetail_c :: [a] -> [a]
safetail_c [] = []
safetail_c xs = Main.tail xs

luhnDouble :: Int -> Int
luhnDouble n = if 2 * n > 9 then 2 * n - 9 else 2 * n

luhn :: Int -> Int -> Int -> Int -> Bool
luhn a b c d = (luhnDouble a + b + luhnDouble c + d) `mod` 10 == 0
```

```haskell
λ> halve [1,2,3,4,5,6]
([1,2,3],[4,5,6])
λ> halve [1,2,3,4,5]
([1,2],[3,4,5])
λ> third_a [1,2,3,4,5]
3
λ> third_b [1,2,3,4,5]
3
λ> third_c [1,2,3,4,5]
3
λ> safetail_a [1,2,3]
[2,3]
λ> safetail_a []
[]
λ> safetail_b [1,2,3]
[2,3]
λ> safetail_b []
[]
λ> safetail_c [1,2,3]
[2,3]
λ> safetail_c []
[]
λ> luhnDouble 3
6
λ> luhnDouble 6
3
λ> luhn 1 7 8 4
True
λ> luhn 4 7 8 3
False
```

```elisp
(defun halve (xs)
  (splitAt (/ (length xs) 2) xs))
(halve '(1 2 3 4 5 6))
((1 2 3) (4 5 6))
(halve '(1 2 3 4 5))
((1 2) (3 4 5))

(defun third_a (xs)
  (caddr xs))
(third_a '(1 2 3 4 5))
3
(defun third_b (xs)
  (nth 2 xs))
(third_b '(1 2 3 4 5))
3
(defun third_c (xs)
  (pcase xs
    (`(,f ,s ,x . ,rest) x)))
(third_c '(1 2 3 4 5))
3

(defun safetail_a (xs)
  (if (eq xs '()) xs (tail/pcase xs)))
(safetail_a '(1 2 3))
(2 3)
(safetail_a '())
nil
(eq '() nil)
t
(defun safetail_b (xs)
  (cond ((eq xs '()) xs)
        (t (tail/pcase xs))))
safetail_b
(safetail_b '(1 2 3))
(2 3)
(safetail_b '())
nil
(defun safetail_c (xs)
  (pcase xs
    ('() xs)
    (_ (tail/pcase xs))))
(safetail_c '(1 2 3))
(2 3)
(safetail_c '())
nil
```

**Note:** The empty list is equivalent to `nil`.

```elisp
(defun luhnDouble (n)
  (if (> (* 2 n) 9) (- (* 2 n) 9) (* 2 n)))
(luhnDouble 6)
3
(luhnDouble 3)
6
(defun luhn (a b c d)
  (= (% (+ (luhnDouble a) b (luhnDouble c) d) 10) 0))
(luhn 1 7 8 4)
t
(luhn 4 7 8 3)
nil
```

```rust
>> fn halve(xs: &[i32]) -> (&[i32],&[i32]) { splitAt(xs.len() / 2, xs) }
>> halve(&[1,2,3,4,5,6])
([1, 2, 3], [4, 5, 6])
>> halve(&[1,2,3,4,5])
([1, 2], [3, 4, 5])
>> fn third_a(xs: &[i32]) -> i32 { *head_match(tail_match(tail_match(xs))) }
>> third_a(&[1,2,3,4,5])
3
>> fn third_b(xs: &[i32]) -> i32 { xs[2] }
>> third_b(&[1,2,3,4,5])
3
>> fn third_c(xs: &[i32]) -> i32 { match xs { [_, _, x, ..] => *x, _ => panic!() } }
>> third_c(&[1,2,3,4,5])
3
```

**Note 1:** Since the function takes a reference, in order to return a value we
need to dereference the borrow.

**Note 2:** For `third_c` we need to define what happens with lists that has less than
`3` elements. The shortest solution is to use a wildcard.

```rust
>> fn safetail_a(xs: &[i32]) -> &[i32] { if xs.len() == 0 { xs } else { tail_match(xs) } }
>> safetail_a(&[1,2,3])
[2, 3]
>> safetail_a(&[])
[]
```

**Note 1:** The `tail_match` was already defined in a "safe" way, however in
general `tail` can panic for an empty list.

**Note 2:** Since we express both "guarded equations" and "patter matching" in
Rust via the `match` expressions, both `safetail_b` and `safetail_c` would look
almost exactly as `tail_match`, so they are not defined.

```rust
>> fn luhnDouble(n: i32) -> i32 { if 2*n > 9 { 2*n-9 } else { 2*n } }
>> luhnDouble(3)
6
>> luhnDouble(6)
3
>> fn luhn(a: i32, b: i32, c: i32, d: i32) -> bool { (luhnDouble(a) + b + luhnDouble(c) + d) % 10 == 0 }
>> luhn(1,7,8,4)
true
>> luhn(4,7,8,3)
false
```
