# 7 Higher-Order functions

## 7.1 Basic concepts

```haskell
add :: Int -> Int -> Int
add x y = x + y

addL :: Int -> (Int -> Int)
addL = \x -> (\y -> x + y)

λ> add 1 2
3
λ> addL 1 2
3

twice :: (a -> a) -> a -> a
twice f x = f (f x)

λ> twice (*2) 3
12
λ> twice reverse [1,2,3]
[1,2,3]
```

```elisp
(defun add (x y)
  (+ x y))
(add 1 2)
3

(defun addL (x)
  (lambda (y) (+ x y)))
(addL 1)
(closure ((x . 1)) (y) (+ x y))
((addL 1) 2) ;; does not work
(defalias 'addL1 (addL 1))
(addL1 2)
3

(defun twice (f x)
  (funcall f (funcall f x)))
(twice 'addL1 3)
5
(twice (apply-partially '* 2) 3)
12
(twice 'reverse '(1 2 3))
(1 2 3)
```

## 7.2 Processing lists

```haskell
mapL :: (a -> b) -> [a] -> [b]
mapL f xs = [f x | x <- xs]

λ> mapL (+1) [1,3,5,7]
[2,4,6,8]

λ> mapL even [1,2,3,4]
[False,True,False,True]

λ> mapL reverse ["abc","def","ghi"]
["cba","fed","ihg"]

λ> mapL (mapL (+1)) [[1,2,3],[4,5]]
[[2,3,4],[5,6]]

mapR :: (a -> b) -> [a] -> [b]
mapR f [] = []
mapR f (x:xs) = f x : mapR f xs

filterL :: (a -> Bool) -> [a] -> [a]
filterL p xs = [x | x <- xs, p x]

λ> filterL even [1..10]
[2,4,6,8,10]

λ> filterL (> 5) [1..10]
[6,7,8,9,10]

λ> filterL (/= ' ') "abc def ghi"
"abcdefghi"

filterR :: (a -> Bool) -> [a] -> [a]
filterR p []                 = []
filterR p (x:xs) | p x       = x : filterR p xs
                 | otherwise = filterR p xs

sumsqreven :: [Int] -> Int
sumsqreven ns = sum (mapL (^2) (filterL even ns))

λ> sumsqreven [1,2,3,4,5]
20

λ> all even [2,4,6,8]
True

λ> any odd [2,4,6,8]
False

λ> takeWhile even [2,4,6,7,8]
[2,4,6]

λ> dropWhile odd [1,3,5,6,7]
[6,7]
```

```elisp
(defun my-map/r (f l)
  (pcase l
    ('() ())
    (`(,x . ,xs) (cons (funcall f x) (my-map/r f xs)))))
(my-map/r 'addL1 '(1 3 5 7))
(2 4 6 8)
(my-map/r 'even '(1 2 3 4))
(nil t nil t)
(my-map/r 'reverse '("abc" "def" "ghi"))
("cba" "fed" "ihg")
(my-map/r (apply-partially 'my-map/r 'addL1) '((1 2 3) (4 5)))
((2 3 4) (5 6))
```

**Note:** Only the recursive version is implemented because there is no
direct equvalent for Haskell generators in Elisp.

```elisp
(require 'dash)

(defun sumsqreven (ns)
  (-sum (my-map/r (lambda (x) (* x x)) (my-filter/r 'even ns))))
(sumsqreven '(1 2 3 4 5))
20

(-all? 'even '(2 4 6 8))
t
(-any 'odd '(2 4 6 8))
nil
(seq-take-while 'even '(2 4 6 7 8))
(2 4 6)
(seq-drop-while 'odd '(1 3 5 6 7))
(6 7)
```

## 7.3 The `foldr` function

```haskell
sumF :: Num a => [a] -> a
sumF = foldr (+) 0

λ> sumF [1,2,3,4,5]
15

productF :: Num a => [a] -> a
productF = foldr (*) 1

λ> productF [1,2,3,4,5]
120

orF :: [Bool] -> Bool
orF = foldr (||) False

λ> orF [True,False,True]
True

andF :: [Bool] -> Bool
andF = foldr (&&) True

λ> andF [True,False,True]
False

foldrR :: (a -> b -> b) -> b -> [a] -> b
foldrR f v [] = v
foldrR f v (x:xs) = f x (foldrR f v xs)

lengthF :: [a] -> Int
lengthF = foldr (\_ n -> 1+n) 0

λ> lengthF [3,2,4,5,1]
5

snoc :: a -> [a] -> [a]
snoc x xs = xs ++ [x]

reverseS :: [a] -> [a]
reverseS [] = []
reverseS (x:xs) = snoc x (reverseS xs)

λ> reverseS [3,2,4,5,1]
[1,5,4,2,3]

reverseF :: [a] -> [a]
reverseF = foldr snoc []
```

```elisp
(defun sum/sr (ns)
  (seq-reduce '+ ns 0))
(sum/sr '(1 2 3 4 5))
15
(defun sum/rf (ns)
  (-reduce-from '+ 0 ns))
(sum/rf '(1 2 3 4 5))
15
(defun sum/rd (ns)
  (-reduce '+ ns))
(sum/fd '(1 2 3 4 5))
15
(apply '+ '(1 2 3 4 5))
15
```

**Note 1:** The direct analog of `foldr` is `seq-reduce`. There is also an
equivalent `-reduce-from` from the `dash` package which only differs in the
order of parameters. However, the implementation of `sum` and the following
functions can use the first element of the list as the "initial value", thus
they can also be implemented via `-reduce`, also from `dash`.

**Note 2:** Since the function `+` in Elisp naturally takes arbitrary number of
arguments, the same result can be achieved by using `apply` with it directly
(the function `apply` uses the last parameter as a list of arguments).

```elisp
(defun product/sr (ns)
  (seq-reduce '* ns 1))
(product/sr '(1 2 3 4 5))
120
(defun product/rf (ns)
  (-reduce-from '* 1 ns))
(product/rf '(1 2 3 4 5))
120
(defun product/rd (ns)
  (-reduce '* ns))
(product/rd '(1 2 3 4 5))
120
(apply '* '(1 2 3 4 5))
120

(defalias 'or/f (lambda (a b) (or a b)))
(defun or/sr (ns)
  (seq-reduce 'or/f ns nil))
(or/sr '(t nil t))
t
(defun or/rf (ns)
  (-reduce-from 'or/f nil ns))
(or/rf '(t nil t))
t
(defun or/rd (ns)
  (-reduce 'or/f ns))
(or/rd '(t nil t))
t
```

**Note:** In Elisp, `or` and `and` are "special forms" because they evaluate
their arguments "lazily". Because of that, neither can be used as a regular
function (including the use with `apply`). We have defined an alias for a two
parameter lambda which calls `or`.

```elisp
(defalias 'and/f (lambda (a b) (and a b)))
(defun and/sr (ns)
  (seq-reduce 'and/f ns t))
(and/sr '(t nil t))
nil
(defun and/rf (ns)
  (-reduce-from 'and/f t ns))
(and/rf '(t nil t))
nil
(defun and/rd (ns)
  (-reduce 'and/f ns))
(and/rd '(t nil t))
nil

(defun foldr/r (f v l)
  (pcase l
    ('() v)
    (`(,x . ,xs) (funcall f x (foldr/r f v xs)))))
(defalias 'sum/frr (lambda (l) (foldr/r '+ 0 l)))
(sum/frr '(1 2 3 4 5))
15

(defalias 'length/f (lambda (l) (foldr/r (lambda (_ n) (+ 1 n)) 0 l)))
(length/f '(3 2 4 5 1))
5

(defun snoc (x xs)
  (append xs (list x)))
(snoc 3 '(1 2))
(1 2 3)

(defalias 'reverse/f (lambda (l) (foldr/r 'snoc nil l)))
(reverse/f '(3 2 4 5 1))
(1 5 4 2 3)
```

## 7.4 The `foldl` function

```haskell
sumL :: Num a => [a] -> a
sumL = sum' 0
       where
          sum' v []     = v
          sum' v (x:xs) = sum' (v+x) xs

lengthL :: [a] -> Int
lengthL = foldl (\n _ -> n+1) 0

reverseL :: [a] -> [a]
reverseL = foldl (\xs x -> x:xs) []

foldlR :: (a -> b -> a) -> a -> [b] -> a
foldlR f v [] = v
foldlR f v (x:xs) = foldlR f (f v x) xs
```

```elisp
(defun sumL (v l)
  (pcase l
    ('() v)
    (`(,x . ,xs) (sumL (+ v x) xs))))
(defun sum/l (l) (sumL 0 l))
(sum/l '(1 2 3 4 5))
15

(defun foldl/r (f v l)
  (pcase l
    ('() v)
    (`(,x . ,xs) (foldl/r f (funcall f v x) xs))))

(defun length/l (l)
  (foldl/r (lambda (n _) (+ n 1)) 0 l))
(length/l '(3 2 4 5 1))
5

(defun reverse/l (l)
  (foldl/r (lambda (xs x) (cons x xs)) nil l))
(reverse/l '(3 2 4 5 1))
(1 5 4 2 3)
```

## 7.5 The composition operator

```haskell
comp :: (b -> c) -> (a -> b) -> (a -> c)
f `comp` g = \x -> f (g x)

compose :: [a -> a] -> (a -> a)
compose = foldr comp id

quadruple :: Num a => a -> a
quadruple = compose [(*2), (*2)]

λ> quadruple 3
12
```

```elisp
(defalias '2* (apply-partially '* 2))
(2* 3)
6

(defun comp (f g)
  (lambda (x) (funcall f (funcall g x))))
(funcall (comp '2* '1+) 1)

(defun id (x) x)
id
(funcall (comp '2* 'id) 3)
6

(defun compose (l)
  (lambda (x) (funcall (-reduce-from 'comp 'id l) x)))

(defalias 'quadruple (compose '(2* 2*)))
(quadruple 3)
12
```

**Note:** Using the function composition function `comp` does not make the
code more readable vs. normal nested function application when need to use
`comp` via `funcall` or `apply`. However, use of `defalias` helps.

## 7.6 Binary string transmitter

`chapter7-functions.hs`
```haskell
import Data.Char

type Bit = Int

bin2intZ :: [Bit] -> Int
bin2intZ bits = sum [w*b | (w,b) <- zip weights bits]
                where weights = iterate (*2) 1

bin2intF :: [Bit] -> Int
bin2intF = foldr (\x y -> x + 2*y) 0

int2bin :: Int -> [Bit]
int2bin 0 = []
int2bin n = n `mod` 2 : int2bin (n `div` 2)

make8 :: [Bit] -> [Bit]
make8 bits = take 8 (bits ++ repeat 0)

encode :: String -> [Bit]
encode = concat . map (make8 . int2bin . ord)

chop8 :: [Bit] -> [[Bit]]
chop8 [] = []
chop8 bits = take 8 bits : chop8 (drop 8 bits)

decode :: [Bit] -> String
decode = map (chr . bin2intF) . chop8

transmit :: String -> String
transmit = decode . channel . encode

channel :: [Bit] -> [Bit]
channel = id
```

```haskell
λ> take 10 (iterate (*2) 1)
[1,2,4,8,16,32,64,128,256,512]
λ> bin2intZ [1,0,1,1]
13
λ> int2bin 13
[1,0,1,1]
λ> make8 (int2bin 13)
[1,0,1,1,0,0,0,0]
λ> encode "abc"
[1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0]
λ> chop8 (encode "abc")
[[1,0,0,0,0,1,1,0],[0,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0]]
λ> decode [1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0]
"abc"
λ> transmit "higher-order functions are easy"
"higher-order functions are easy"
```

```elisp
(defun iterate (f v n)
  (if (= n 0) nil
    (cons v (iterate f (funcall f v) (1- n)))))
(iterate '2* 1 8)
(1 2 4 8 16 32 64 128)

(defun bit2int/i (b)
  (apply '+ (mapcar* '* b (iterate '2* 1 (length b)))))
(bit2int/z '(1 0 1 1))
13

(defun bit2int/f (b)
  (foldr/r (lambda (x y) (+ x (2* y))) 0 b))
(bit2int/f '(1 0 1 1))
13

(defun int2bin (n)
  (if (= n 0) nil
    (cons (% n 2) (int2bin (/ n 2)))))
(int2bin 13)
(1 0 1 1)

(defun make8 (l)
  (seq-take (seq-concatenate 'list l (make-list 8 0)) 8))
(make8 (int2bin 13))
(1 0 1 1 0 0 0 0)

(defun encode (s)
  (apply 'seq-concatenate 'list
         (mapcar (lambda (c) (make8 (int2bin c))) s)))
(encode "abc")
(1 0 0 0 0 1 1 0 0 1 0 0 ...)

(defun chop8 (bits)
  (seq-partition bits 8))
(chop8 (encode "abc"))
((1 0 0 0 0 1 1 0) (0 1 0 0 0 1 1 0) (1 1 0 0 0 1 1 0))

(defun decode (bits)
  (concat (mapcar 'bit2int/f (chop8 bits))))
(decode '(1 0 0 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0))
"abc"

(defalias 'channel 'id)

(defalias 'transmit (compose '(decode channel encode)))
(transmit "higher-order functions are easy")
"higher-order functions are easy"
```

## 7.7 Voting algorithms

`chapter7-functions.hs`
```haskell
import Data.List

votes :: [String]
votes = ["Red", "Blue", "Green", "Blue", "Blue", "Red"]

count :: Eq a => a -> [a] -> Int
count x = length . filter (== x)

rmdups :: Eq a => [a] -> [a]
rmdups [] = []
rmdups (x:xs) = x : filter (/= x) (rmdups xs)

result :: Ord a => [a] -> [(Int,a)]
result vs = sort [(count v vs, v) | v <- rmdups vs]

winner :: Ord a => [a] -> a
winner = snd . last . result

ballots :: [[String]]
ballots = [["Red", "Green"],
           ["Blue"],
           ["Green", "Red", "Blue"],
           ["Blue", "Green", "Red"],
           ["Green"]]

rmempty :: Eq a => [[a]] -> [[a]]
rmempty = filter (/= [])

elim :: Eq a => a -> [[a]] -> [[a]]
elim x = map (filter (/= x))

rank :: Ord a => [[a]] -> [a]
rank = map snd . result . map head

winner' :: Ord a => [[a]] -> a
winner' bs = case rank (rmempty bs) of
              [c] -> c
              (c:cs) -> winner' (elim c bs)
```

```haskell
λ> count "Red" votes
2
λ> rmdups votes
["Red","Blue","Green"]
λ> result votes
[(1,"Green"),(2,"Red"),(3,"Blue")]
λ> winner votes
"Blue"

λ> elim "Red" ballots
[["Green"],["Blue"],["Green","Blue"],["Blue","Green"],["Green"]]
λ> rank ballots
["Red","Blue","Green"]
λ> winner' ballots
"Green"
```

```elisp
(setq votes '("Red" "Blue" "Green" "Blue" "Blue" "Red"))

(defun count/s (x l)
  (seq-count (lambda (s) (string= s x)) l))
(count/s "Red" votes)
2

(defun rmdups (l)
  (pcase l
    ('() ())
    (`(,x . ,xs) (cons x
                       (seq-filter (lambda (s) (not (string= s x)))
                                   (rmdups xs))))))
(rmdups votes)
("Red" "Blue" "Green")

(defun result (vs)
  (seq-sort (lambda (l r) (< (car l) (car r)))
            (seq-map (lambda (v) (cons (count/s v vs) v)) (rmdups vs))))
(result votes)
((1 . "Green") (2 . "Red") (3 . "Blue"))

(defalias 'winner (compose '(cdar last result)))
(winner votes)
"Blue"

(setq ballots '(("Red" "Green")
           ("Blue")
           ("Green" "Red" "Blue")
           ("Blue" "Green" "Red")
           ("Green")))

(defun rmempty (ll) (seq-remove 'not ll))

(defun elim (x ll)
  (seq-map (lambda (l) (seq-filter (lambda (s) (not (string= s x))) l)) ll))
(elim "Red" ballots)
(("Green") ("Blue") ("Green" "Blue") ("Blue" "Green") ("Green"))

(defun rank (ll)
    (seq-map 'cdr (result (seq-map 'car ll))))
(rank ballots)
("Red" "Blue" "Green")

(defun winner/b (bs)
  (pcase (rank (rmempty bs))
    (`(,c . nil) c)
    (`(,c . ,cs) (winner/b (elim c bs)))))
(winner/b ballots)
"Green"
```

## 7.9 Exercises

```haskell
listcomp :: (a -> b) -> (a -> Bool) -> [a] -> [b]
listcomp f p xs = map f (filter p xs)

listcomp2 :: (a -> b) -> (a -> Bool) -> [a] -> [b]
listcomp2 f p = map f . filter p

λ> listcomp2 ord (== 'a') ['a', 'b', 'c', 'a']
[97,97]

allM :: (a -> Bool) -> [a] -> Bool
allM p xs = and (map p xs)

allM2 :: (a -> Bool) -> [a] -> Bool
allM2 p = and . map p

anyM :: (a -> Bool) -> [a] -> Bool
anyM p xs = or (map p xs)

anyM2 :: (a -> Bool) -> [a] -> Bool
anyM2 p = or . map p

takeWhileR :: (a -> Bool) -> [a] -> [a]
takeWhileR _ []                 = []
takeWhileR p (x:xs) | p x       = x : takeWhileR p xs
                    | otherwise = []

dropWhileR :: (a -> Bool) -> [a] -> [a]
dropWhileR _ [] = []
dropWhileR p (x:xs) | p x       = dropWhileR p xs
                    | otherwise = x : xs

mapF :: (a -> b) -> [a] -> [b]
mapF f = foldr cf []
          where cf x xs = f x : xs

mapFL :: (a -> b) -> [a] -> [b]
mapFL f = foldr (\x xs -> f x : xs) []

filterF :: (a -> Bool) -> [a] -> [a]
filterF p = foldr cp []
             where cp x xs | p x       = x : xs
                           | otherwise = xs

filterFL :: (a -> Bool) -> [a] -> [a]
filterFL p = foldr (\x xs -> if p x then x : xs else xs) []

dec2int :: [Int] -> Int
dec2int = foldl (\x y -> 10*x + y) 0

λ> dec2int [2,3,4,5]
2345

myCurry :: ((a,b) -> c) -> (a -> b -> c)
myCurry f = \x -> (\y -> f (x,y))

myCurry2 :: ((a,b) -> c) -> (a -> b -> c)
myCurry2 f = \x y -> f (x,y)

plus :: Num a => (a,a) -> a
plus (x,y) = x + y

λ> plus (2,3)
5
λ> myCurry plus 2 3
5

myUncurry :: (a -> b -> c) -> ((a,b) -> c)
myUncurry f = \(x,y) -> f x y

λ> myUncurry add (2,3)
5

unfold :: (a -> Bool) -> (a -> b) -> (a -> a) -> a -> [b]
unfold p h t x | p x       = []
               | otherwise = h x : unfold p h t (t x)

int2binU :: Int -> [Bit]
int2binU = unfold (== 0) (`mod` 2) (`div` 2)

chop8U :: [Bit] -> [[Bit]]
chop8U = unfold null (take 8) (drop 8)

iterateU :: (a -> a) -> a -> [a]
iterateU f = unfold (const False) id f

mapU :: (a -> b) -> [a] -> [b]
mapU f = unfold null (f . head) tail

addParity :: [Bit] -> [Bit]
addParity bits = bits ++ [(count 1 bits) `mod` 2]
λ> addParity [1,0,1]
[1,0,1,0]
λ> addParity [1,1,1]
[1,1,1,1]

encodeP :: String -> [Bit]
encodeP = concat . map (addParity . make8 . int2bin . ord)
λ> encodeP "abc"
[1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0]

chopN :: Int -> [Bit] -> [[Bit]]
chopN _ [] = []
chopN n bits = take n bits : chopN n (drop n bits)

chop9 = chopN 9

removeParity :: [Bit] -> [Bit]
removeParity bits = if addParity bits' == bits then bits' else error "Parity error"
                    where bits' = init bits
λ> removeParity [1,0,1,0]
[1,0,1]
λ> removeParity [1,0,1,1]
Main: Parity error

decodeP :: [Bit] -> String
decodeP = map (chr . bin2intF . removeParity) . chop9

faultyChannel = tail

λ> (decodeP . channel . encodeP) "abc"
"abc"
λ> (decodeP . faultyChannel . encodeP) "abc"
Main: Parity error

interleave :: a -> a -> [a]
interleave x y = [x, y] ++ interleave x y

altMap :: (a -> b) -> (a -> b) -> [a] -> [b]
altMap f g = zipWith ($) (interleave f g)

λ> altMap (+10) (+100) [0,1,2,3,4]
[10,101,12,103,14]
```

**Note:** The symbol `$` denotes "function application operator": `f $ x = f x`.

```haskell
luhnDouble :: Int -> Int
luhnDouble n = if 2 * n > 9 then 2 * n - 9 else 2 * n

luhn :: [Int] -> Bool
luhn ns = sum (altMap luhnDouble id ns) `mod` 10 == 0

λ> luhn [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
True

int2dec :: Int -> [Int]
int2dec n = reverse (unfold (== 0) (`mod` 10) (`div` 10) n)

λ> int2dec 1111222233334444
[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
λ> luhn (int2dec 1111222233334444)
True
```

```elisp
(defun listcomp (f p xs)
  (seq-map f (seq-filter p xs)))
(listcomp '2* (lambda (x) (eq ?a x)) "abca")
(194 194)

(defun listcomp/2 (f p)
  (compose (list (apply-partially 'seq-map f)
                 (apply-partially 'seq-filter p))))
(funcall (listcomp/2 '2* (lambda (x) (eq ?a x))) "abca")
(194 194)

(defun all/m (p xs)
  (eval (cons 'and (seq-map p xs))))
(all/m 'even '(2 4 6 8))
t

(defun any/m (p xs)
  (eval (cons 'or (seq-map p xs))))
(any/m 'odd '(2 4 6 8))
nil
```

**Note:** Use of `eval` is a way to apply the special forms `and`, `or` to
a list of arguments. The call to `cons` creates a cons cell for the
function call expression.

```elisp
(defun all/m (p xs)
  (eval (cons 'and (seq-map p xs))))
(all/m 'even '(2 4 6 8))
t

(defun any/m (p xs)
  (eval (cons 'or (seq-map p xs))))
(any/m 'odd '(2 4 6 8))
nil

(defun take-while (p l)
  (pcase l
    ('() '())
    (`(,x . ,xs) (if (funcall p x) (cons x (take-while p xs)) '()))))
(take-while 'even '(2 4 6 7 8))
(2 4 6)

(defun drop-while (p l)
  (pcase l
    ('() '())
    (`(,x . ,xs) (if (funcall p x) (drop-while p xs) l))))
(drop-while 'odd '(1 3 5 6 7))
(6 7)

(defun map/fl (f l)
  (foldr/r (lambda (x xs) (cons (funcall f x) xs)) '() l))
(map/fl '1+ '(1 2 3 4 5))
(2 3 4 5 6)

(defun filter/fl (p l)
  (foldr/r (lambda (x xs) (if (funcall p x) (cons x xs) xs)) '() l))
(filter/fl 'odd '(1 2 3 4 5))
(1 3 5)

(defun dec2int (d)
  (foldl/r (lambda (x y) (+ (* 10 x) y)) 0 d))
(dec2int '(2 3 4 5))
2345

(defun curry (f)
  (lambda (x) (lambda (y) (funcall f x y))))
(funcall (funcall (curry #'+) 2) 3)
5

(defun uncurry (f)
  (lambda (x y) (funcall (funcall f x) y)))
(funcall (uncurry (curry #'+)) 2 3)
5

(defun unfold (p h t x)
  (if (funcall p x) '()
    (cons (funcall h x) (unfold p h t (funcall t x)))))

(defalias 'int2bin/u (apply-partially 'unfold 'zerop
                             (lambda (x) (mod x 2))
                             (lambda (x) (/ x 2))))
(int2bin/u 13)
(1 0 1 1)

(defalias 'chop8/u (apply-partially 'unfold 'null
                                    (apply-partially 'take 8)
                                    (apply-partially 'nthcdr 8)))
(chop8/u (encode "abc"))
((1 0 0 0 0 1 1 0) (0 1 0 0 0 1 1 0) (1 1 0 0 0 1 1 0))

(defun map/u (f x) (unfold 'null
                           (lambda (x) (funcall f (car x)))
                           'cdr x))
(map/u #'1+ '(1 2 3 4 5))
(2 3 4 5 6)

(defun add-parity (bits)
  (append bits (list (% (seq-count (apply-partially '= 1) bits) 2))))
(add-parity '(1 0 1))
(1 0 1 0)
(add-parity '(1 1 1))
(1 1 1 1)

(defun encode/p (s)
  (apply 'seq-concatenate 'list
         (mapcar (lambda (c) (add-parity (make8 (int2bin c)))) s)))
(encode/p "abc")
(1 0 0 0 0 1 1 0 1 0 1 0 ...)

(defun chop9 (bits)
  (seq-partition bits 9))

(defun remove-parity (bits)
  (let ((raw-bits (take (1- (length bits)) bits)))
    (if (equal (add-parity raw-bits) bits) raw-bits
      (error "Parity error"))))
(remove-parity '(1 0 1 0))
(1 0 1)
(remove-parity '(1 0 1 1))
Debugger entered--Lisp error: (error "Parity error")

(defun decode/p (bits)
  (concat (mapcar (lambda (ch) (bit2int/f (remove-parity ch))) (chop9 bits))))
(decode/p (encode/p "abc"))

(defalias 'faulty-channel #'cdr)

(funcall (compose '(decode/p channel encode/p)) "abc")
"abc"
(funcall (compose '(decode/p faulty-channel encode/p)) "abc")
Debugger entered--Lisp error: (error "Parity error")

(defun altmap (f g l)
  (pcase l
    ('() '())
    (`(,x . ,xs) (cons (funcall f x) (altmap g f xs)))))
(altmap (apply-partially #'+ 10) (apply-partially #'+ 100) '(0 1 2 3 4))
(10 101 12 103 14)

(defun luhnDouble (n)
  (if (> (* 2 n) 9) (- (* 2 n) 9) (* 2 n)))

(require 'dash)

(defun luhn (ns)
  (= (mod (-sum (altmap 'luhnDouble 'id ns)) 10) 0))
(luhn '(1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4))
t

(defun int2dec (n)
  (nreverse (unfold 'zerop (lambda (x) (mod x 10)) (lambda (x) (/ x 10)) n)))
(int2dec 1111222233334444)
(1 1 1 1 2 2 2 2 3 3 3 3 ...)
(luhn (int2dec 1111222233334444))
t
```
