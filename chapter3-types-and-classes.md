# 3. Types and classes

# 3.2 Basic types

### Logical values

Haskell: `Bool` type, `False` and `True` values
Elisp: `booleanp` test predicate, `nil` and `t` values (no need to quote)
Rust: `bool` type, `false` and `true` values

### Single characters

Haskell: `Char` type, Unicode
Elisp: `characterp` test predicate, Unicode (superset). Represented by an integer
       (the character code). A character constant uses leading `?`: `?A`.
Rust: `char` type, Unicode, stored as 32-bit word

### Strings of characters

Haskell: `String` type
Elisp: arrays of characters, fixed length, `stringp` test predicate
Rust: `str` type, equivalent to an array of bytes, must contain valid UTF-8

### Fixed-precision integers

Haskell: `Int` type, 64-bit
Elisp: `fixnump`, "fixnums", the range depends on the machine word size
Rust: `i..` and `u..` types with specified width, `i|usize` for machine word sized
      The type `integer` can be printed by the compiler, however it's not a valid
      type name.

### Arbitraty-precision integers

Haskell: `Integer` type
Elisp: `bignump`, "bignums"
Rust: only available as a non-standard package `rug`, `rug::Integer`

### Single-precision floating-point numbers

Haskell: `Float` type
Elisp: none
Rust: `f32`

### Double-precision floating-point numbers

Haskell: `Double` type
Elisp: `floatp`
Rust: `f64`

## 3.3 List types

Haskell: `[]`, any length
Elisp: `'()` for lists, `[]` for arrays. Arrays are fixed size, with constant
       element access time. Arrays, lists and strings are generalized as "sequences".
Rust: arrays `[]` (fixed size) and vectors `std::Vec` (dynamic)

## 3.4 Tuple types

Haskell: `()`, length must be known because arity is part of the type
Elisp: also lists, see https://www.codeproject.com/Articles/1186940/Lisps-Mysterious-Tuple-Problem
       However, for pairs, the "cons cell" representation is often used.
Rust: `()`

## 3.6 Curried functions

Haskell has a convention to define multi-parameter functions via currying:

```haskell
λ> add_c :: Int -> Int -> Int ; add_c a b = a+b
λ> add_c 2 3
5
λ> add_c 2
add_c 2 :: Int -> Int
```

Defining `add` as a function of multiple parameters (in the C-like style) will
in fact define it as a function on a tuple:

```haskell
λ> add_m :: (Int,Int) -> Int ; add_m (a,b) = a+b
λ> add_m (2,3)
5
```

Unlike Haskell, both Elisp and Rust do not define multi-parameter functions
via currying. That means, partial application is only possible by creating
closures.

Elisp has a helper function `apply-partially` which returns a new function
that has some arguments bound to the specified values. It is possible to
return a function from a function, however there is no automatic conversion
of a multi-parameter function into a set of currying functions.

```elisp
(defun add_m (a b) (+ a b))
add_m
(add_m 2 3)
5

(defalias 'add_m_2 (apply-partially 'add_m 2))
add_m_2
(add_m_2 3)
5
```

```rust
>> fn add_m (a: u32, b: u32) -> u32 { a + b }
>> add_m(2,3)
5
>> { let add_m_2 = |x| add_m(2,x); add_m_2(3) }
5
```

**Note:** Braces are needed for `evcxr` in order to avoid attempting capturing
the closure as a value.

## 3.7 Polymorphic types

Since Haskell only allows using names starting with low case for variables,
putting such a name into a context where type name is expected makes it a
type variable—a polymorphic type.

```haskell
λ> id :: a -> a ; id x = x
λ> id 0
0
λ> id [1, 2, 3]
[1,2,3]
λ> id "hello"
"hello"
```

Elisp does not specify types, thus all functions parameters are polymorphic
by default. It's the function's responsibility to check the actual type
of each argument and raise an exception if the type is wrong.

```elisp
(defun id (x) x)
id
(id 0)
0
(id '(1 2 3))
(1 2 3)
(id "hello")
"hello"
```

Rust allows lambdas without parameter type specification, however the type is
inferred on the first application and can not be changed later:

```rust
>> { let id = |x| x; id(0); id([1, 2, 3]) }
[E0308] Error: mismatched types
   ╭─[command:1:1]
   │
 1 │ { let id = |x| x; id(0); id([1, 2, 3]) }
   │             ┬            ─┬ ────┬────
   │             ╰────────────────────────── note: closure parameter defined here
   │                           │     │
   │                           ╰──────────── arguments to this function are incorrect
   │                                 │
   │                                 ╰────── expected integer, found `[{integer}; 3]`
───╯
```

Rust allows type parameters for functions:

```rust
>> fn id<T>(x: &T) -> &T { x }
>> id(&0)
0
>> id(&[1,2,3])
[1, 2, 3]
>> id(&"hello")
"hello"
```

**Note:** we have to use a reference type in order to avoid making a copy.

## 3.8 Overloaded types

In Haskell, a function can be defined for a class of types, or for several
classes. This is called "overloading." This isn't the same overloading approach
as in C++ where functions taking different types of parameters can share a name.
A proper explanation of overloading in Haskell requires mentioning types. With
the simplest example it may appear that overloading just limits the application
domain of a function.

```haskell
λ> flip :: Num a => a -> a; flip n = -n
λ> flip 5
-5
λ> flip "hello"
*** Error: No instance for (Num String) arising from a use of ‘flip’
```

In Elisp, there are no types specified for parameters, hence overloading can only
be achieved in the function implementation.

In Rust, "traits" are similar to classes in Haskell, and they are used for the
same purpose. There is also a way to limit application of a function to certain
types, similar to the example.

```rust
>> fn flip<T: std::ops::Neg<Output=T>>(n: T) -> T { -n }
>> flip(5)
-5
>> flip("hello")
[E0277] Error: the trait bound `&str: Neg` is not satisfied
```

## 3.9 Basic classes

### Equality types

Class `Eq`. Defines operations `==` and `/=`. Not supported for functions.
Float `NaN` is an instance of `Eq` but it is not equal to itself (by IEEE 754
definition).

```haskell
λ> False == False
True
λ> 'a' == 'b'
False
λ> "abc" == "abc"
True
λ> 'a' /= 'b'
True
λ> [1,2,3] == [1,2,3]
True
λ> acos 2 == acos 2
False
λ> acos 2 /= acos 2
True
```

Elisp is more low level and contains several "equality" predicates. The
predicate `eq` checks if two objects are the same object, while `equals` checks
contents.  Note that `equals` returns `t` for `NaN`. There are also "numerical
comparison" functions `=` and `/=` (work only for numbers) and the predicate
`eql` which works like `eq` but compares numbers both by type and value.

Characters can be compared using `char-equal`, and strings using `string=`
(aliased to `string-equal`).

```elisp
(eq nil nil)
t
(eq ?a ?b)
nil
(eq "abc" "abc")
nil
(equal "abc" "abc")
t
(string= "abc" "abc")
t
(eq '+ '+)
t
(eq '(1 2 3) '(1 2 3))
nil
(equal '(1 2 3) '(1 2 3))
t
(eq [1 2 3] [1 2 3])
nil
(equal [1 2 3] [1 2 3])
t
(eq (acos 2) (acos 2))
nil
(equal (acos 2) (acos 2))
t
(= (acos 2) (acos 2))
nil
(eql (acos 2) (acos 2))
t
(eq 1 1.0)
nil
(eql 1 1.0)
nil
(= 1 1.0)
t
(equal 1 1.0)
nil
```

In Rust, there is a trait `std::cmp::PartialEq` which defines a non-reflexive
comparison class. Due to the rule for NaN comparison, floats belong to this
class. The `Eq` class dervices from `PartialEq`, adding reflexivity requirement.
Comparison operators are `==` and `!=`.

```rust
>> false == false
true
>> 'a' == 'b'
false
>> "abc" == "abc"
true
>> 'a' != 'b'
true
>> [1,2,3] == [1,2,3]
true
>> f64::acos(2.0) == f64::acos(2.0)
false
>> f64::acos(2.0) != f64::acos(2.0)
true
```

### Ordered types

The `Ord` class in Haskell derives from `Eq` and adds `6` operations:
4 comparison and `min`, `max`.

```haskell
λ> False < True
True
λ> min 'a' 'b'
'a'
λ> "elegant" < "elephant"
True
λ> [1,2,3] < [1,2]
False
λ> ('a',2) < ('b',1)
True
λ> ('a',2) < ('a',1)
False
```

In Elisp, the same operations are implemented for numbers. For strings, there
are `string<`, `string-lessp` and `string-greaterp`. For lists, there does not
seem to be a built-in comparison besides equality. Also not possible to compare
`nil` with `t`.

```elisp
(min ?a ?b)
97
(string< "elegant" "elephant")
t
```

Rust provides `PartialOrd` trait which defines the comparison operators. Note
that since `PartialOrd` does not relate to `Eq`, it is not requred that
`a <= a`. The trait `Ord` is defined as `Eq + PartialOrd`, and adds `max`,
`min` and `clamp` functions.

Arrays of the same type and size can be compared. Vectors implement `PartialOrd`
and can be compared by calling corresponding methods.

```rust
>> false < true
true
>> std::cmp::min('a', 'b')
'a'
>> "elegant" < "elephant"
true
>> [1,2] < [1,3]
true
>> vec![1,2,3].lt(&vec![1,2])
false
>> ('a',2) < ('b',1)
true
>> ('a',2) < ('a',1)
false
```

### Showable types

The class `Show` defines a function `show` for converting a value into a
string. The printed strings are inteded to be readable by the `read` function.

```haskell
λ> show False
"False"
λ> show 'a'
"'a'"
λ> show 123
"123"
λ> show [1,2,3]
"[1,2,3]"
λ> show ('a',False)
"('a',False)"
λ> show "hello"
"\"hello\""
```

Elisp has functions for printing values in human-readable and machine-readable
forms.

```elisp
(prin1-to-string nil)
"nil"
(prin1-to-string ?a)
"97"
(prin1-to-string 123)
"123"
(prin1-to-string '(1 2 3))
"(1 2 3)"
(prin1-to-string [1 2 3])
"[1 2 3]"
(prin1-to-string '(?a nil))
"(97 nil)"
(prin1-to-string "hello")
"\"hello\""
```

In Rust, the trait is called `ToString` and the method is `to_string`.  These
methods can not be applied directly to arrays and vectors, while tuples do not
implement `to_string` at all.

```rust
>> false.to_string()
"false"
>> 'a'.to_string()
"a"
>> 123.to_string()
"123"
>> "hello".to_string()
"hello"
```

### Readable types

In Haskell, the intended type needs to be specified. All values rendered into a
string can be read back.

```haskell
λ> read "False" :: Bool
False
λ> read "'a'" :: Char
'a'
λ> read "123" :: Int
123
λ> read "[1,2,3]" :: [Int]
[1,2,3]
λ> read "('a',False)" :: (Char,Bool)
('a',False)
λ> read "\"hello\"" :: String
"hello"
```

In Elisp, `read-from-string` returns parsed value and the position in the string
to start the next read.

```elisp
(read-from-string "nil")
(nil . 3)
(read-from-string "97")
(97 . 2)
(read-from-string "123")
(123 . 3)
(read-from-string "123 456")
(123 . 3)
(read-from-string "(1 2 3)")
((1 2 3) . 7)
(read-from-string "[1 2 3]")
([1 2 3] . 7)
(read-from-string "(97 nil)")
((97 nil) . 8)
(read-from-string "\"hello\"")
("hello" . 7)
```

Rust classes provide `parse` function which returns an instance of `Err`
containing both the result and the parsed value in case of success.

```rust
>> "false".parse::<bool>().unwrap()
false
>> "a".parse::<char>().unwrap()
'a'
>> "123".parse::<u32>().unwrap()
123
>> "hello".parse::<String>().unwrap()
"hello"
```

### Numeric types

In Haskell, the `Num` class provides `3` arithmetic operations (no division),
as well as `negate`, `abs`, and `signum` functions.

```haskell
λ> 1+2
3
λ> (+) 1 2
3
λ> 1.0+2.0
3.0
λ> 1.0+2
3.0
λ> negate 3.0
-3.0
λ> abs (-3)
3
λ> signum (-3)
-1
```

**Note:** Negative numbers have to be parenthesized otherwise due to priority
`abs` is applied to the `-` function first.

Elisp lacks built-in `signum` function.

```elisp
(+ 1 2)
3
(+ 1.0 2.0)
3.0
(+ 1.0 2)
3.0
(- 3.0)
-3.0
(abs -3)
3
```

Rust defines separate traits for each arithmetic operation that allow defining
them for all needed types. Operations between instances of different types must
be explicitly defined (for example, adding duration value to time).  For numeric
types, there are no implementations of operations for mixed types, and all
operands must be be explicitly coerced.

```rust
>> 1+2
3
>> 1.0+2.0
3.0
>> 1.0+(2 as f64)
3.0
>> -3.0
-3.0
>> (-3.0_f64).abs()
3.0
>> (-3.0_f64).signum()
-1.0
```

### Integral types

The class `Integral` is a subclass of `Num` and provides `div` and `mod` methods.

```haskell
λ> 7 `div` 2
3
λ> 7 `mod` 2
1
λ> div 7 2
3
```

As mentioned in **Chapter 2**, Elisp uses integer division when operands are
integers.

```elisp
(/ 7 2)
3
(% 7 2)
1
```

**Note:** There is also a `mod` function in Elisp which implements the algebraic
modulo operation. The difference between these functions can be seen when
working with negative arguments:

```elisp
(% -7 2)
-1
(mod -7 2)
1
```

```rust
>> 7 / 2
3
>> 7 % 2
1
```

### Fractional types

In Haskell, the `Fractional` class defines floating point division, and `recip`
method for finding the reciprocal.

```haskell
λ> 7.0 / 2.0
3.5
λ> 7 / 2
3.5
λ> recip 2.0
0.5
λ> recip 2
0.5
```

In Elisp, the same `/` function is used for both. However, finding the
reciprocal for integers does not make sense.

```elisp
(/ 7.0 2.0)
3.5
(/ 7.0 2)
3.5
(/ 2.0)
0.5
(/ 2)
0
```

Rust defines `recip` method for float types.

```rust
>> 7.0 / 2.0
3.5
>> (2.0_f64).recip()
0.5
```
