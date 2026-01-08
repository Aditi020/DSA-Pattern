# Complete DSA & Java Collections Cheatsheet

## 📚 Table of Contents
1. [📊 Java Collections Overview](#java-collections-overview)
2. [🎯 Core Collections Reference](#core-collections-reference)
3. [🔢 Array Patterns](#array-patterns)
4. [🔤 String Patterns](#string-patterns)
5. [📚 Stack, Queue & Linked List](#stack-queue-linked-list-patterns)
6. [🎯 Advanced Collections & Data Structures](#advanced-collections-data-structures)
7. [🔍 Searching Algorithms](#searching-algorithms)
8. [🌳 Tree Patterns](#tree-patterns)
9. [🕸️ Graph Patterns](#graph-patterns)
10. [🔑 Hashing Patterns](#hashing-patterns)
11. [🌀 Recursion & Backtracking](#recursion-backtracking-patterns)
12. [⚡ Dynamic Programming](#dynamic-programming-patterns)
13. [🤏 Greedy Algorithms](#greedy-algorithms)
14. [💡 Problem-Solving Framework](#problem-solving-framework)
15. [🚀 Quick Reference](#quick-reference)

---

## 📊 Java Collections Overview

> **Key Insight:** 90%+ of coding problems can be solved with Java Collections

### **Essential Collections Coverage Matrix**
| Collection Type | Coverage | Best For |
|----------------|----------|----------|
| **ArrayList/HashMap** | 80% | Most problems |
| **HashSet/Stack/Queue** | 85% | Unique elements, LIFO/FIFO |
| **PriorityQueue/TreeMap** | 90% | Ordering, heaps, ranges |
| **All Collections Combined** | 95%+ | Virtually all problems |

---

## 🎯 Core Collections Reference

### **1. String Manipulation - StringBuilder**
```java
StringBuilder sb = new StringBuilder();
```
| Method | Time | Use Case |
|--------|------|----------|
| `sb.append(value)` | O(1) | Add to end |
| `sb.reverse()` | O(n) | Reverse string |
| `sb.insert(pos, val)` | O(n) | Insert at position |
| `sb.delete(start, end)` | O(n) | Remove substring |
| `sb.charAt(i)` | O(1) | Get character |
| `sb.setCharAt(i, c)` | O(1) | Set character |

### **2. Dynamic Arrays - ArrayList**
```java
List<Integer> list = new ArrayList<>();
```
| Operation | Method | Time |
|-----------|--------|------|
| **Add** | `list.add(element)` | O(1)* |
| **Access** | `list.get(index)` | O(1) |
| **Search** | `list.contains(element)` | O(n) |
| **Remove** | `list.remove(index)` | O(n) |
| **Sort** | `Collections.sort(list)` | O(n log n) |

### **3. Fast Lookups - HashMap (Most Important!)**
```java
Map<K, V> map = new HashMap<>();
```
| Pattern | Code Snippet |
|---------|-------------|
| **Frequency Counting** | `map.put(key, map.getOrDefault(key, 0) + 1)` |
| **Two Sum** | Check `complement = target - nums[i]` |
| **Grouping** | `map.computeIfAbsent(key, k -> new ArrayList<>()).add(value)` |

### **4. Ordered Data - TreeMap**
```java
TreeMap<Integer, String> tm = new TreeMap<>();
```
| Method | Returns | Use Case |
|--------|---------|----------|
| `tm.ceilingKey(k)` | Smallest key ≥ k | Next available |
| `tm.floorKey(k)` | Largest key ≤ k | Previous available |
| `tm.subMap(from, to)` | Keys in range | Range queries |

### **5. Heaps - PriorityQueue**
```java
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
```
| Pattern | Implementation |
|---------|----------------|
| **K Largest** | Min heap, remove when size > K |
| **K Smallest** | Max heap, remove when size > K |
| **Median** | Two heaps (max left, min right) |

---

## 🔢 Array Patterns

### **1. Sliding Window**
```java
// Fixed Size Window (K)
int windowSum = 0;
for(int i = 0; i < k; i++) windowSum += arr[i];
maxSum = windowSum;

for(int i = k; i < n; i++) {
    windowSum += arr[i] - arr[i-k];
    maxSum = Math.max(maxSum, windowSum);
}

// Variable Size Window
int left = 0, maxLen = 0;
Map<Integer, Integer> map = new HashMap<>();

for(int right = 0; right < n; right++) {
    // Add arr[right] to window
    while(/* condition invalid */) {
        // Remove arr[left] from window
        left++;
    }
    maxLen = Math.max(maxLen, right - left + 1);
}
```
**Problems:** Maximum Sum Subarray of Size K, Longest Substring Without Repeating Characters

### **2. Two Pointers**
```java
// Pair Sum
int left = 0, right = nums.length - 1;
while(left < right) {
    int sum = nums[left] + nums[right];
    if(sum == target) return new int[]{left, right};
    else if(sum < target) left++;
    else right--;
}

// Dutch National Flag (3-way partition)
int low = 0, mid = 0, high = nums.length - 1;
while(mid <= high) {
    if(nums[mid] == 0) swap(nums, low++, mid++);
    else if(nums[mid] == 1) mid++;
    else swap(nums, mid, high--);
}
```
**Problems:** Pair with Given Sum, Trapping Rain Water, Sort Colors

### **3. Kadane's Algorithm (Max Subarray Sum)**
```java
int maxSoFar = nums[0], maxEndingHere = nums[0];
for(int i = 1; i < nums.length; i++) {
    maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
    maxSoFar = Math.max(maxSoFar, maxEndingHere);
}
return maxSoFar;
```

### **4. Prefix Sum**
```java
int[] prefix = new int[n+1];
for(int i = 0; i < n; i++) {
    prefix[i+1] = prefix[i] + nums[i];
}
// Sum from i to j = prefix[j+1] - prefix[i]
```
**Problems:** Subarray Sum Equals K, Range Sum Query

---

## 🔤 String Patterns

### **1. String Sliding Window**
```java
// Longest Substring Without Repeating Characters
Map<Character, Integer> map = new HashMap<>();
int left = 0, maxLen = 0;

for(int right = 0; right < s.length(); right++) {
    char c = s.charAt(right);
    if(map.containsKey(c)) {
        left = Math.max(left, map.get(c) + 1);
    }
    map.put(c, right);
    maxLen = Math.max(maxLen, right - left + 1);
}
```

### **2. Palindrome Patterns**
```java
// Expand Around Center
private int expandAroundCenter(String s, int left, int right) {
    while(left >= 0 && right < s.length() 
          && s.charAt(left) == s.charAt(right)) {
        left--;
        right++;
    }
    return right - left - 1;
}
```

### **3. Anagrams & Frequency Counting**
```java
// Check if two strings are anagrams
int[] count = new int[26];
for(char c : s1.toCharArray()) count[c-'a']++;
for(char c : s2.toCharArray()) count[c-'a']--;
for(int val : count) if(val != 0) return false;
return true;

// Group Anagrams
Map<String, List<String>> map = new HashMap<>();
for(String str : strs) {
    char[] chars = str.toCharArray();
    Arrays.sort(chars);
    String key = new String(chars);
    map.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
}
return new ArrayList<>(map.values());
```

### **4. String DP Patterns**
```java
// Longest Common Subsequence
int[][] dp = new int[m+1][n+1];
for(int i = 1; i <= m; i++) {
    for(int j = 1; j <= n; j++) {
        if(s1.charAt(i-1) == s2.charAt(j-1)) {
            dp[i][j] = 1 + dp[i-1][j-1];
        } else {
            dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
        }
    }
}
```

---

## 📚 Stack, Queue & Linked List Patterns

### **1. Stack Patterns**
```java
// Monotonic Stack - Next Greater Element
Stack<Integer> stack = new Stack<>();
int[] result = new int[n];
Arrays.fill(result, -1);

for(int i = 0; i < n; i++) {
    while(!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
        result[stack.pop()] = nums[i];
    }
    stack.push(i);
}

// Valid Parentheses
Stack<Character> stack = new Stack<>();
for(char c : s.toCharArray()) {
    if(c == '(') stack.push(')');
    else if(c == '[') stack.push(']');
    else if(c == '{') stack.push('}');
    else if(stack.isEmpty() || stack.pop() != c) return false;
}
return stack.isEmpty();
```

### **2. Queue Patterns**
```java
// BFS with Queue
Queue<TreeNode> queue = new LinkedList<>();
queue.offer(root);
while(!queue.isEmpty()) {
    int size = queue.size();
    for(int i = 0; i < size; i++) {
        TreeNode node = queue.poll();
        // Process node
        if(node.left != null) queue.offer(node.left);
        if(node.right != null) queue.offer(node.right);
    }
}

// Circular Queue Implementation
class MyCircularQueue {
    private int[] data;
    private int head, tail, size, capacity;
    
    public boolean enQueue(int value) {
        if(isFull()) return false;
        data[tail] = value;
        tail = (tail + 1) % capacity;
        size++;
        return true;
    }
}
```

### **3. Linked List Patterns**
```java
// Reverse Linked List
ListNode reverse(ListNode head) {
    ListNode prev = null, curr = head;
    while(curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// Detect Cycle (Floyd's Algorithm)
ListNode slow = head, fast = head;
while(fast != null && fast.next != null) {
    slow = slow.next;
    fast = fast.next.next;
    if(slow == fast) return true; // Cycle detected
}
return false;

// Merge Two Sorted Lists
ListNode dummy = new ListNode(0);
ListNode curr = dummy;
while(l1 != null && l2 != null) {
    if(l1.val < l2.val) {
        curr.next = l1;
        l1 = l1.next;
    } else {
        curr.next = l2;
        l2 = l2.next;
    }
    curr = curr.next;
}
curr.next = (l1 != null) ? l1 : l2;
```

---

## 🎯 Advanced Collections & Data Structures

### **1. Trie Implementation**
```java
class TrieNode {
    TrieNode[] children = new TrieNode[26];
    boolean isEnd;
}

class Trie {
    private TrieNode root;
    
    public void insert(String word) {
        TrieNode node = root;
        for(char c : word.toCharArray()) {
            if(node.children[c-'a'] == null) {
                node.children[c-'a'] = new TrieNode();
            }
            node = node.children[c-'a'];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for(char c : word.toCharArray()) {
            if(node.children[c-'a'] == null) return false;
            node = node.children[c-'a'];
        }
        return node.isEnd;
    }
}
```

### **2. Union-Find (Disjoint Set)**
```java
class UnionFind {
    private int[] parent, rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for(int i = 0; i < n; i++) parent[i] = i;
    }
    
    public int find(int x) {
        if(parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    
    public boolean union(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if(rootX == rootY) return false;
        
        if(rank[rootX] < rank[rootY]) parent[rootX] = rootY;
        else if(rank[rootX] > rank[rootY]) parent[rootY] = rootX;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        return true;
    }
}
```

### **3. Segment Tree (Range Sum Query)**
```java
class SegmentTree {
    private int[] tree;
    private int n;
    
    public SegmentTree(int[] nums) {
        n = nums.length;
        tree = new int[4 * n];
        build(nums, 0, 0, n-1);
    }
    
    private void build(int[] nums, int node, int start, int end) {
        if(start == end) tree[node] = nums[start];
        else {
            int mid = (start + end) / 2;
            build(nums, 2*node+1, start, mid);
            build(nums, 2*node+2, mid+1, end);
            tree[node] = tree[2*node+1] + tree[2*node+2];
        }
    }
}
```

---

## 🔍 Searching Algorithms

### **Binary Search Patterns**
```java
// Standard Binary Search
int binarySearch(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) return mid;
        else if(nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Find First Occurrence
int firstOccurrence(int[] nums, int target) {
    int left = 0, right = nums.length - 1, result = -1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) {
            result = mid;
            right = mid - 1; // Continue searching left
        } else if(nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return result;
}

// Search in Rotated Sorted Array
int searchRotated(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) return mid;
        
        // Left half is sorted
        if(nums[left] <= nums[mid]) {
            if(nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } 
        // Right half is sorted
        else {
            if(nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```

---

## 🌳 Tree Patterns

### **1. Tree Traversals**
```java
// Inorder (Left, Root, Right) - BST gives sorted order
void inorder(TreeNode root) {
    if(root == null) return;
    inorder(root.left);
    System.out.print(root.val + " ");
    inorder(root.right);
}

// Preorder (Root, Left, Right) - Copy tree
// Postorder (Left, Right, Root) - Delete tree

// Level Order (BFS)
List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if(root == null) return result;
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    
    while(!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        
        for(int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            currentLevel.add(node.val);
            if(node.left != null) queue.offer(node.left);
            if(node.right != null) queue.offer(node.right);
        }
        result.add(currentLevel);
    }
    return result;
}
```

### **2. BST Operations**
```java
// Validate BST
boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

boolean validate(TreeNode node, long min, long max) {
    if(node == null) return true;
    if(node.val <= min || node.val >= max) return false;
    return validate(node.left, min, node.val) 
        && validate(node.right, node.val, max);
}

// Kth Smallest Element
int kthSmallest(TreeNode root, int k) {
    Stack<TreeNode> stack = new Stack<>();
    while(true) {
        while(root != null) {
            stack.push(root);
            root = root.left;
        }
        root = stack.pop();
        if(--k == 0) return root.val;
        root = root.right;
    }
}
```

### **3. Tree Algorithms**
```java
// Lowest Common Ancestor
TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if(root == null || root == p || root == q) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if(left != null && right != null) return root;
    return left != null ? left : right;
}

// Diameter of Binary Tree
int diameter = 0;
int diameterOfBinaryTree(TreeNode root) {
    height(root);
    return diameter;
}

int height(TreeNode node) {
    if(node == null) return 0;
    int left = height(node.left);
    int right = height(node.right);
    diameter = Math.max(diameter, left + right);
    return 1 + Math.max(left, right);
}
```

---

## 🕸️ Graph Patterns

### **1. Graph Representations**
```java
// Adjacency List
List<List<Integer>> adj = new ArrayList<>();
for(int i = 0; i < n; i++) adj.add(new ArrayList<>());

// Add edge u -> v
adj.get(u).add(v);

// For weighted graphs
List<List<int[]>> adj = new ArrayList<>();
adj.get(u).add(new int[]{v, weight});
```

### **2. Graph Traversals**
```java
// DFS
void dfs(int node, boolean[] visited, List<List<Integer>> adj) {
    visited[node] = true;
    for(int neighbor : adj.get(node)) {
        if(!visited[neighbor]) {
            dfs(neighbor, visited, adj);
        }
    }
}

// BFS
void bfs(int start, List<List<Integer>> adj) {
    boolean[] visited = new boolean[adj.size()];
    Queue<Integer> queue = new LinkedList<>();
    queue.offer(start);
    visited[start] = true;
    
    while(!queue.isEmpty()) {
        int node = queue.poll();
        for(int neighbor : adj.get(node)) {
            if(!visited[neighbor]) {
                visited[neighbor] = true;
                queue.offer(neighbor);
            }
        }
    }
}
```

### **3. Shortest Path Algorithms**
```java
// Dijkstra's Algorithm
int[] dijkstra(int n, List<List<int[]>> adj, int src) {
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    
    PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> a[1]-b[1]);
    pq.offer(new int[]{src, 0});
    
    while(!pq.isEmpty()) {
        int[] curr = pq.poll();
        int u = curr[0], d = curr[1];
        
        if(d > dist[u]) continue;
        
        for(int[] edge : adj.get(u)) {
            int v = edge[0], w = edge[1];
            if(dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.offer(new int[]{v, dist[v]});
            }
        }
    }
    return dist;
}
```

---

## 🔑 Hashing Patterns

### **1. Subarray Problems with HashMap**
```java
// Subarray Sum Equals K
int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 1); // Base case: sum 0 seen once
    
    int sum = 0, count = 0;
    for(int num : nums) {
        sum += num;
        if(map.containsKey(sum - k)) {
            count += map.get(sum - k);
        }
        map.put(sum, map.getOrDefault(sum, 0) + 1);
    }
    return count;
}

// Longest Subarray with Sum Divisible by K
int longestSubarray(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1); // remainder 0 at index -1
    
    int sum = 0, maxLen = 0;
    for(int i = 0; i < nums.length; i++) {
        sum += nums[i];
        int remainder = ((sum % k) + k) % k; // Handle negative
        
        if(map.containsKey(remainder)) {
            maxLen = Math.max(maxLen, i - map.get(remainder));
        } else {
            map.put(remainder, i);
        }
    }
    return maxLen;
}
```

---

## 🌀 Recursion & Backtracking Patterns

### **1. Combinatorial Problems**
```java
// Generate All Subsets
List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), result);
    return result;
}

void backtrack(int[] nums, int start, List<Integer> path, 
               List<List<Integer>> result) {
    result.add(new ArrayList<>(path));
    
    for(int i = start; i < nums.length; i++) {
        path.add(nums[i]);
        backtrack(nums, i + 1, path, result);
        path.remove(path.size() - 1);
    }
}

// N-Queens
List<List<String>> solveNQueens(int n) {
    List<List<String>> result = new ArrayList<>();
    char[][] board = new char[n][n];
    for(char[] row : board) Arrays.fill(row, '.');
    backtrack(board, 0, result);
    return result;
}

void backtrack(char[][] board, int row, List<List<String>> result) {
    if(row == board.length) {
        result.add(construct(board));
        return;
    }
    
    for(int col = 0; col < board.length; col++) {
        if(isValid(board, row, col)) {
            board[row][col] = 'Q';
            backtrack(board, row + 1, result);
            board[row][col] = '.';
        }
    }
}
```

---

## ⚡ Dynamic Programming Patterns

### **1. Knapsack Patterns**
```java
// 0/1 Knapsack
int knapsack(int[] values, int[] weights, int capacity) {
    int n = values.length;
    int[][] dp = new int[n + 1][capacity + 1];
    
    for(int i = 1; i <= n; i++) {
        for(int w = 1; w <= capacity; w++) {
            if(weights[i-1] <= w) {
                dp[i][w] = Math.max(
                    dp[i-1][w],
                    values[i-1] + dp[i-1][w - weights[i-1]]
                );
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][capacity];
}
```

### **2. String DP Patterns**
```java
// Longest Common Subsequence
int lcs(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m+1][n+1];
    
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            if(s1.charAt(i-1) == s2.charAt(j-1)) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}

// Edit Distance
int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m+1][n+1];
    
    for(int i = 0; i <= m; i++) dp[i][0] = i;
    for(int j = 0; j <= n; j++) dp[0][j] = j;
    
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            if(word1.charAt(i-1) == word2.charAt(j-1)) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + Math.min(
                    dp[i-1][j],    // delete
                    Math.min(
                        dp[i][j-1],    // insert
                        dp[i-1][j-1]   // replace
                    )
                );
            }
        }
    }
    return dp[m][n];
}
```

### **3. Grid DP Patterns**
```java
// Unique Paths
int uniquePaths(int m, int n) {
    int[][] dp = new int[m][n];
    
    for(int i = 0; i < m; i++) dp[i][0] = 1;
    for(int j = 0; j < n; j++) dp[0][j] = 1;
    
    for(int i = 1; i < m; i++) {
        for(int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
```

---

## 🤏 Greedy Algorithms

### **1. Interval Problems**
```java
// Merge Intervals
int[][] merge(int[][] intervals) {
    if(intervals.length <= 1) return intervals;
    
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    List<int[]> result = new ArrayList<>();
    int[] current = intervals[0];
    result.add(current);
    
    for(int[] interval : intervals) {
        if(interval[0] <= current[1]) {
            current[1] = Math.max(current[1], interval[1]);
        } else {
            current = interval;
            result.add(current);
        }
    }
    return result.toArray(new int[result.size()][]);
}

// Activity Selection
int maxMeetings(int[] start, int[] end) {
    int n = start.length;
    int[][] meetings = new int[n][2];
    for(int i = 0; i < n; i++) meetings[i] = new int[]{start[i], end[i]};
    
    Arrays.sort(meetings, (a, b) -> Integer.compare(a[1], b[1]));
    
    int count = 1, lastEnd = meetings[0][1];
    for(int i = 1; i < n; i++) {
        if(meetings[i][0] >= lastEnd) {
            count++;
            lastEnd = meetings[i][1];
        }
    }
    return count;
}
```

---

## 💡 Problem-Solving Framework

### **1. Step-by-Step Approach**
1. **Understand** - Restate problem in own words
2. **Examples** - Create test cases (edge cases!)
3. **Brute Force** - Start with naive solution
4. **Optimize** - Identify bottlenecks, use patterns
5. **Implement** - Write clean, modular code
6. **Test** - Run through all test cases
7. **Analyze** - Discuss time/space complexity

### **2. Pattern Recognition Flowchart**
```
Problem → Identify Type → Choose Pattern → Implement
    ↓
Array/String
    ├── Need subarray/substring? → Sliding Window
    ├── Need pairs? → Two Pointers/HashMap
    ├── Need max/min sum? → Kadane/Prefix Sum
    └── Need ordering? → Sorting
    
Tree/Graph
    ├── Need traversal? → BFS/DFS
    ├── Need shortest path? → Dijkstra/BFS
    └── Need connectivity? → Union-Find
    
Optimization
    ├── Need top K? → Heap
    ├── Need range queries? → Segment Tree
    └── Need DP? → Memoization/Tabulation
```

### **3. Complexity Analysis Cheatsheet**
| Algorithm | Time | Space |
|-----------|------|-------|
| Sliding Window | O(n) | O(k) |
| Two Pointers | O(n) | O(1) |
| BFS/DFS | O(V+E) | O(V) |
| Binary Search | O(log n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n) avg | O(log n) |
| DP | O(n*m) | O(n*m) |

---

## 🚀 Quick Reference

### **Most Frequently Used Methods**
```java
// Collections
Collections.sort(list);
Collections.reverse(list);
Collections.max(list);
Collections.min(list);
Collections.frequency(list, element);

// Arrays
Arrays.sort(arr);
Arrays.fill(arr, value);
Arrays.copyOf(arr, length);
Arrays.equals(arr1, arr2);
Arrays.binarySearch(sortedArr, key);

// Strings
str.charAt(i);
str.substring(i, j);
str.split(regex);
str.trim();
str.toLowerCase();
str.toUpperCase();

// Math
Math.max(a, b);
Math.min(a, b);
Math.abs(x);
Math.pow(a, b);
Math.sqrt(x);
```

### **Time Complexity by Operation**
| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| ArrayList | O(1) | O(n) | O(1)* | O(n) |
| LinkedList | O(n) | O(n) | O(1) | O(1) |
| HashMap | N/A | O(1) | O(1) | O(1) |
| TreeMap | N/A | O(log n) | O(log n) | O(log n) |
| Heap | N/A | N/A | O(log n) | O(log n) |
| Stack | O(1) top | O(n) | O(1) | O(1) |
| Queue | O(1) front | O(n) | O(1) | O(1) |

*Amortized O(1)

---

## 📌 Final Tips

1. **Master these patterns** - They cover 90%+ of interview questions
2. **Practice recognizing patterns** - More important than memorizing code
3. **Use Java Collections** - Don't reinvent the wheel
4. **Start with brute force** - Then optimize
5. **Test edge cases** - Empty, single element, duplicates, large inputs
6. **Communicate your thinking** - Interviewers care about process

---
