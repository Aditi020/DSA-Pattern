# **Java Streams Cheatsheet**
## **1. Creation Streams**
```java
// From Collections
list.stream()              // Sequential stream
list.parallelStream()      // Parallel processing

// From Arrays
Arrays.stream(array)
Arrays.stream(array, start, end)

// From values
Stream.of(1, 2, 3)
Stream.iterate(0, n -> n + 1)    // Infinite stream
Stream.generate(() -> Math.random())  // Random infinite

// Range (IntStream/LongStream)
IntStream.range(0, 10)           // 0-9 (exclusive)
IntStream.rangeClosed(0, 10)     // 0-10 (inclusive)
```

## **2. Intermediate Operations (Lazy)**
```java
// Filtering
.filter(x -> x > 5)              // Keep if true
.distinct()                      // Remove duplicates
.limit(n)                        // First n elements
.skip(n)                         // Skip first n elements

// Transformation
.map(x -> x * 2)                 // Transform each element
.flatMap(list -> list.stream())  // Flatten nested lists
.mapToInt(x -> x)                // Convert to primitive stream

// Sorting
.sorted()                        // Natural order
.sorted((a,b) -> b - a)          // Custom comparator
```

## **3. Terminal Operations (Eager)**
```java
// Collection/Reduction
.collect(Collectors.toList())    // To List
.collect(Collectors.toSet())     // To Set
.toArray()                       // To array
.reduce(0, (a,b) -> a + b)       // Reduce to single value
.count()                         // Count elements

// Searching
.findFirst()                     // Optional of first element
.findAny()                       // Any element (parallel)
.anyMatch(x -> x > 5)            // True if any matches
.allMatch(x -> x > 0)            // True if all match
.noneMatch(x -> x < 0)           // True if none match

// Aggregation
.min(Comparator.naturalOrder())  // Optional min
.max(Comparator.naturalOrder())  // Optional max
.sum()                           // Sum (Int/Long/DoubleStream)
.average()                       // Average (OptionalDouble)
```

## **4. Collectors - Most Useful for Coding**
```java
// To Collections
Collectors.toList()
Collectors.toSet()
Collectors.toMap(keyMapper, valueMapper)
Collectors.toCollection(ArrayList::new)

// Grouping & Partitioning (VERY USEFUL)
Collectors.groupingBy(x -> x.length())  // Group by length
Collectors.groupingBy(
    keyMapper, 
    Collectors.toList()                 // Downstream collector
)

Collectors.partitioningBy(x -> x > 5)   // True/False partitions

// Aggregation
Collectors.counting()                   // Count elements
Collectors.summingInt(x -> x)           // Sum
Collectors.averagingInt(x -> x)         // Average
Collectors.summarizingInt(x -> x)       // Stats: count,sum,min,max,avg

// Joining Strings
Collectors.joining()                    // "abc"
Collectors.joining(", ")                // "a, b, c"
Collectors.joining(", ", "[", "]")      // "[a, b, c]"
```

## **5. Practical Patterns for Coding Problems**

### **5.1 Frequency Counting (Most Common)**
```java
// Character frequency
Map<Character, Long> freq = s.chars()
    .mapToObj(c -> (char)c)
    .collect(Collectors.groupingBy(
        c -> c, 
        Collectors.counting()
    ));

// Word frequency
Map<String, Long> wordFreq = Arrays.stream(sentence.split(" "))
    .collect(Collectors.groupingBy(
        word -> word, 
        Collectors.counting()
    ));
```

### **5.2 Filter & Transform**
```java
// Get even numbers squared
List<Integer> result = numbers.stream()
    .filter(n -> n % 2 == 0)
    .map(n -> n * n)
    .collect(Collectors.toList());

// Get strings starting with 'A'
List<String> aWords = words.stream()
    .filter(w -> w.startsWith("A"))
    .collect(Collectors.toList());
```

### **5.3 Find Min/Max**
```java
// Find max in list
Optional<Integer> max = numbers.stream()
    .max(Integer::compare);

// Find min string by length
Optional<String> shortest = words.stream()
    .min(Comparator.comparingInt(String::length));
```

### **5.4 Check Conditions**
```java
// Check if all positive
boolean allPositive = numbers.stream()
    .allMatch(n -> n > 0);

// Check if any negative
boolean hasNegative = numbers.stream()
    .anyMatch(n -> n < 0);
```

### **5.5 Reduce Operations**
```java
// Sum of squares
int sumOfSquares = numbers.stream()
    .map(n -> n * n)
    .reduce(0, Integer::sum);

// String concatenation
String concatenated = words.stream()
    .reduce("", (a, b) -> a + b);
```

### **5.6 Sorting & Limiting**
```java
// Get top 3 largest numbers
List<Integer> top3 = numbers.stream()
    .sorted(Collections.reverseOrder())
    .limit(3)
    .collect(Collectors.toList());

// Get distinct sorted
List<Integer> distinctSorted = numbers.stream()
    .distinct()
    .sorted()
    .collect(Collectors.toList());
```

### **5.7 Array/List Conversion**
```java
// Int array to List
List<Integer> list = Arrays.stream(arr)
    .boxed()  // IntStream to Stream<Integer>
    .collect(Collectors.toList());

// List to int array
int[] array = list.stream()
    .mapToInt(Integer::intValue)
    .toArray();
```

### **5.8 Grouping Patterns (Very Powerful)**
```java
// Group strings by length
Map<Integer, List<String>> byLength = words.stream()
    .collect(Collectors.groupingBy(String::length));

// Group and count
Map<String, Long> countByWord = words.stream()
    .collect(Collectors.groupingBy(
        word -> word, 
        Collectors.counting()
    ));

// Group and get max
Map<String, Optional<Integer>> maxByGroup = list.stream()
    .collect(Collectors.groupingBy(
        obj -> obj.category,
        Collectors.mapping(
            obj -> obj.value,
            Collectors.maxBy(Integer::compare)
        )
    ));
```

### **5.9 FlatMap (Nested Collections)**
```java
// Flatten list of lists
List<List<Integer>> nested = ...;
List<Integer> flat = nested.stream()
    .flatMap(List::stream)
    .collect(Collectors.toList());

// Word characters
List<Character> chars = words.stream()
    .flatMap(word -> word.chars().mapToObj(c -> (char)c))
    .collect(Collectors.toList());
```

## **6. Performance Tips**
```java
// Use primitive streams for performance
IntStream.range(0, n)                     // Faster than Stream<Integer>
    .sum();                               // No boxing/unboxing

// Parallel for large data
list.parallelStream()                     // For 10k+ elements
    .filter(...)
    .collect(...);

// Chain operations efficiently
// Bad: multiple streams
long count1 = list.stream().filter(...).count();
long count2 = list.stream().filter(...).count();

// Good: single stream with collectors
Map<Boolean, Long> counts = list.stream()
    .collect(Collectors.partitioningBy(
        condition,
        Collectors.counting()
    ));
```

## **7. Common Problem Solutions**

### **Two Sum Variant (Find pairs)**
```java
// Find all pairs that sum to target
List<int[]> pairs = IntStream.range(0, nums.length)
    .boxed()
    .flatMap(i -> IntStream.range(i+1, nums.length)
        .filter(j -> nums[i] + nums[j] == target)
        .mapToObj(j -> new int[]{nums[i], nums[j]})
    )
    .collect(Collectors.toList());
```

### **Anagram Groups**
```java
// Group anagrams
Map<String, List<String>> anagramGroups = words.stream()
    .collect(Collectors.groupingBy(
        word -> {
            char[] chars = word.toCharArray();
            Arrays.sort(chars);
            return new String(chars);
        }
    ));
```

### **Top K Frequent Elements**
```java
// Using streams for top K
List<Integer> topK = nums.stream()
    .collect(Collectors.groupingBy(
        n -> n, 
        Collectors.counting()
    ))
    .entrySet().stream()
    .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
    .limit(k)
    .map(Map.Entry::getKey)
    .collect(Collectors.toList());
```

## **8. When to Use Streams**

### **Use Streams When:**
- Processing collections with multiple operations
- Need concise, readable data transformations
- Working with large data (parallel streams)
- Functional style preferred

### **Avoid Streams When:**
- Need to break early from loops (use regular loops)
- Complex state management required
- Performance-critical small loops
- Need to modify original collection

## **9. Quick Reference Card**

```java
// Most common patterns:
.collect(Collectors.toList())              // → List
.collect(Collectors.groupingBy(...))       // → Grouped Map
.filter(condition).map(transformation)     // → Filter & transform
.sorted().limit(k)                         // → Top K
.anyMatch/allMatch/noneMatch(condition)    // → Boolean checks
.reduce(0, (a,b) -> a+b)                   // → Summation
```

**Time Complexity:** Most stream operations are O(n) for sequential, O(n/p) for parallel with p processors.

**Memory:** Streams are lazy (except terminal ops) and don't create intermediate collections unless necessary.

---
