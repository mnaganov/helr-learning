# 7 Higher-Order functions

## 7.1 Basic concepts

```haskell
add :: Int -> Int -> Int
add x y = x + y

addL :: Int -> (Int -> Int)
addL = \x -> (\y -> x + y)

> add 1 2
3
> addL 1 2
3

twice :: (a -> a) -> a -> a
twice f x = f (f x)

> twice (*2) 3
12
> twice reverse [1,2,3]
[1,2,3]
```

## 7.2 Processing lists

```haskell
mapL :: (a -> b) -> [a] -> [b]
mapL f xs = [f x | x <- xs]

> mapL (+1) [1,3,5,7]
[2,4,6,8]

> mapL even [1,2,3,4]
[False,True,False,True]

> mapL reverse ["abc","def","ghi"]
["cba","fed","ihg"]

> mapL (mapL (+1)) [[1,2,3],[4,5]]
[[2,3,4],[5,6]]

mapR :: (a -> b) -> [a] -> [b]
mapR f [] = []
mapR f (x:xs) = f x : mapR f xs

filterL :: (a -> Bool) -> [a] -> [a]
filterL p xs = [x | x <- xs, p x]

> filterL even [1..10]
[2,4,6,8,10]

> filterL (> 5) [1..10]
[6,7,8,9,10]

> filterL (/= ' ') "abc def ghi"
"abcdefghi"

filterR :: (a -> Bool) -> [a] -> [a]
filterR p []                 = []
filterR p (x:xs) | p x       = x : filterR p xs
                 | otherwise = filterR p xs

sumsqreven :: [Int] -> Int
sumsqreven ns = sum (mapL (^2) (filterL even ns))

> sumsqreven [1,2,3,4,5]
20

> all even [2,4,6,8]
True

> any odd [2,4,6,8]
False

> takeWhile even [2,4,6,7,8]
[2,4,6]

> dropWhile odd [1,3,5,6,7]
[6,7]
```

## 7.3 The `foldr` function

```haskell
sumF :: Num a => [a] -> a
sumF = foldr (+) 0

> sumF [1,2,3,4,5]
15

productF :: Num a => [a] -> a
productF = foldr (*) 1

> productF [1,2,3,4,5]
120

orF :: [Bool] -> Bool
orF = foldr (||) False

> orF [True,False,True]
True

andF :: [Bool] -> Bool
andF = foldr (&&) True

> andF [True,False,True]
False

foldrR :: (a -> b -> b) -> b -> [a] -> b
foldrR f v [] = v
foldrR f v (x:xs) = f x (foldrR f v xs)

lengthF :: [a] -> Int
lengthF = foldr (\_ n -> 1+n) 0

> lengthF [3,2,4,5,1]
5

snoc :: a -> [a] -> [a]
snoc x xs = xs ++ [x]

reverseS :: [a] -> [a]
reverseS [] = []
reverseS (x:xs) = snoc x (reverseS xs)

> reverseS [3,2,4,5,1]
[1,5,4,2,3]

reverseF :: [a] -> [a]
reverseF = foldr snoc []
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

## 7.5 The composition operator

```haskell
comp :: (b -> c) -> (a -> b) -> (a -> c)
f `comp` g = \x -> f (g x)

compose :: [a -> a] -> (a -> a)
compose = foldr comp id

quadruple :: Num a => a -> a
quadruple = compose [(*2), (*2)]

> quadruple 3
12
```

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
> take 10 (iterate (*2) 1)
[1,2,4,8,16,32,64,128,256,512]
> bin2intZ [1,0,1,1]
13
> int2bin 13
[1,0,1,1]
> make8 (int2bin 13)
[1,0,1,1,0,0,0,0]
> encode "abc"
[1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0]
> chop8 (encode "abc")
[[1,0,0,0,0,1,1,0],[0,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0]]
> decode [1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0]
"abc"
> transmit "higher-order functions are easy"
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
> count "Red" votes
2
> rmdups votes
["Red","Blue","Green"]
> result votes
[(1,"Green"),(2,"Red"),(3,"Blue")]
> winner votes
"Blue"

> elim "Red" ballots
[["Green"],["Blue"],["Green","Blue"],["Blue","Green"],["Green"]]
> rank ballots
["Red","Blue","Green"]
> winner' ballots
"Green"
```

## 7.9 Exercises

```haskell
listcomp :: (a -> b) -> (a -> Bool) -> [a] -> [b]
listcomp f p xs = map f (filter p xs)

listcomp2 :: (a -> b) -> (a -> Bool) -> [a] -> [b]
listcomp2 f p = map f . filter p

> listcomp2 ord (== 'a') ['a', 'b', 'c', 'a']
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

> dec2int [2,3,4,5]
2345

myCurry :: ((a,b) -> c) -> (a -> b -> c)
myCurry f = \x -> (\y -> f (x,y))

myCurry2 :: ((a,b) -> c) -> (a -> b -> c)
myCurry2 f = \x y -> f (x,y)

plus :: Num a => (a,a) -> a
plus (x,y) = x + y

> plus (2,3)
5
> myCurry plus 2 3
5

myUncurry :: (a -> b -> c) -> ((a,b) -> c)
myUncurry f = \(x,y) -> f x y

> myUncurry add (2,3)
5

unfold :: (a -> Bool) -> (a -> b) -> (a -> a) -> a -> [b]
unfold p h t x | p x       = []
               | otherwise = h x : unfold p h t (t x)

int2binU :: Int -> [Bit]
int2binU = unfold (== 0) (`mod` 2) (`div` 2)

chop8U :: [Bit] -> [[Bit]]
chop8U = unfold (== []) (take 8) (drop 8)

iterateU :: (a -> a) -> a -> [a]
iterateU f = unfold (const False) id f

mapU :: (a -> b) -> [a] -> [b]
mapU f = unfold null (f . head) tail

addParity :: [Bit] -> [Bit]
addParity bits = bits ++ [(count 1 bits) `mod` 2]
> addParity [1,0,1]
[1,0,1,0]
> addParity [1,1,1]
[1,1,1,1]

encodeP :: String -> [Bit]
encodeP = concat . map (parity . make8 . int2bin . ord)
> encodeP "abc"
[1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0]

chopN :: Int -> [Bit] -> [[Bit]]
chopN _ [] = []
chopN n bits = take n bits : chopN n (drop n bits)

chop9 = chopN 9

removeParity :: [Bit] -> [Bit]
removeParity bits = if addParity bits' == bits then bits' else error "Parity error"
                    where bits' = init bits
> removeParity [1,0,1,0]
[1,0,1]
> removeParity [1,0,1,1]
Main: Parity error

decodeP :: [Bit] -> String
decodeP = map (chr . bin2intF . removeParity) . chop9

faultyChannel = tail

> (decodeP . channel . encodeP) "abc"
"abc"
> (decodeP . faultyChannel . encodeP) "abc"
Main: Parity error

interleave :: a -> a -> [a]
interleave x y = [x, y] ++ interleave x y

altMap :: (a -> b) -> (a -> b) -> [a] -> [b]
altMap f g = zipWith ($) (interleave f g)

> altMap (+10) (+100) [0,1,2,3,4]
[10,101,12,103,14]
```

**Note:** The symbol `$` denotes "function application operator": `f $ x = f x`.

```haskell
luhnDouble :: Int -> Int
luhnDouble n = if 2 * n > 9 then 2 * n - 9 else 2 * n

luhn :: [Int] -> Bool
luhn ns = sum (altMap luhnDouble id ns) `mod` 10 == 0

> luhn [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
True

int2dec :: Int -> [Int]
int2dec n = reverse (unfold (== 0) (`mod` 10) (`div` 10) n)

> int2dec 1111222233334444
[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
> luhn (int2dec 1111222233334444)
True
```
