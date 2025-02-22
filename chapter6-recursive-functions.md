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
of the functions from the standard prelude.

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
