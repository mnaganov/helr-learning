even :: Integral a => a -> Bool
even n = n `mod` 2 == 0

splitAt :: Int -> [a] -> ([a],[a])
splitAt n xs = (take n xs, drop n xs)

recip :: Fractional a => a -> a
recip n = 1/n

abs_if :: Int -> Int
abs_if n = if n >= 0 then n else -n

signum_if :: Int -> Int
signum_if n = if n < 0 then -1 else
                if n == 0 then 0 else 1

abs_grd :: Int -> Int
abs_grd n | n >= 0    = n
          | otherwise = -n

signum_grd :: Int -> Int
signum_grd n | n < 0     = -1
             | n == 0    = 0
             | otherwise = 1

not :: Bool -> Bool
not False = True
not True = False

and :: Bool -> Bool -> Bool
True `and` b = b
False `and` _ = False

fst :: (a,b) -> a
fst (x,_) = x

snd :: (a,b) -> b
snd (_,y) = y

head :: [a] -> a
head (x:_) = x

tail :: [a] -> [a]
tail (_:xs) = xs

add_l :: Int -> (Int -> Int)
add_l = \x -> (\y -> x + y)

const_l :: a -> (b -> a)
const_l x = \_ -> x

odds_l :: Int -> [Int]
odds_l n = map (\x -> x*2 + 1) [0..n-1]

sum :: [Int] -> Int
sum = foldl (+) 0

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
