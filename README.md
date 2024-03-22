# 动态规划
## 45. 跳跃游戏 II
方法一：动态规划

定义：f(i) 为从 0 跳跃到 i 处的最少跳跃次数。

利用向后更新策略，来优化代码的可读性，即从当前 第 i 个位置，去更新后面的 i + j 这个位置的最少跳跃次数。

代码如下：
```c++
int jump(vector<int>& nums) {
    int f[10100], n = nums.size() - 1;
    memset(f, 0x3f, sizeof f);
    f[0] = 0;
    for (int i = 0; i < n; i ++ ) 
        for (int j = 1; j <= nums[i]; j ++ ) 
            if (i + j <= n - 1)
                f[i + j] = min(f[i + j], f[i] + 1);
        
    return f[n - 1];

}
```
方法二：`贪心选择 + 惰性更新`
我们每次都选择从 上一次位置 pre 在 [pre ~ curR] 区间内能够跳跃地最远的那个点 作为选择点。那么我们采用惰性更新来简化代码的复杂度：
```c++
int jump(vector<int>& nums) {
    int ans = 0, curR = 0, curMax = 0, n = nums.size();
    for (int i = 0; i < n - 1; i ++ ) {
        curMax = max(curMax, nums[i] + i);
        if (i == curR) ans ++ , curR = curMax;
    }
    return ans;
}
```

## 91. 解码方法
此题的边界条件和 `LCR 165. 解密数字` 有所不同，我们特别要注意，下标的处理，及其边界的处理。

代码如下
```c++
int numDecodings(string s) {
    int n = s.size();
    int f[n + 1]; memset(f, 0, sizeof f); f[0] = 1;
    for (int i = 1; i <= n; i ++ ) {
        if (s[i - 1] != '0') f[i] += f[i - 1];
        if (i - 1 >= 1) {
            string sub = s.substr(i - 2, 2);
            if (sub <= "26" && sub >= "10") f[i] += f[i - 2];
        }
    }
    return f[n];
}
```

## 139. 单词拆分

总结一个考虑下标的步骤：
1. 先在 1...n 上的坐标进行思考状态方程
2. 然后如果要使用 s[] 时，我们将坐标减 1 即可。
```c++
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> S(wordDict.begin(), wordDict.end());
    int n = s.size();
    int f[n + 1]; memset(f, 0, sizeof f); f[0] = 1;
    for (int i = 1; i <= n; i ++ )
        for (auto e : S) {
            int len = e.size();
            if (i - len >= 0) {
                string sub = s.substr(i - len, len);
                if (sub == e) f[i] |= f[i - len];
            }
        }
    return f[n];
}
```

## 357. 统计各位数字都不同的数字个数
数位 DP 问题，我们这里来总结一下 `特殊与一般` 的关系

首先，如果只有一位数字，那么他应该被人为定义为特殊的数字，而如果有两位数以上则是一般性整数，可以用组合数学来处理

## LeetCode 376. 摆动序列
题意：给定一个数组 nums[] 要找到一个最长的子序列，使得他这个子序列满足摆动的条件。

一、数学思想：

1 > `动态规划 + 状态机`，类似于`股票买卖 + 最长递增数组`的结合，即设计两种状态：一个是以 i 为结尾，最后一个元素是下降状态的最长摆动序列，一个是以 i 为结尾，最后一个元素是上升状态的最长摆动序列。然后就可以`以 O(n^2) 的双重循环`写出这题了

2 > 贪心：即删除最长摆动序列中`除了峰点和谷点`的其他所有元素，如下图所示:
```
                o                       o
            x       x            x
        x
    x                  x        x
o                           o
```
可以看到我们只保留了峰点和谷点，其他在这中间的元素全部删去了。这里我们不去严格证明其正确性，但是从直观上来看，这个做法就是正确的

二、代码实现：

1 > 动态规划：略

2 > 贪心：我们`不去`真正的删除这些中间值元素，而是利用 `分类讨论` 来分别给出 `[0...i]` 的最长末尾上升状态长度 up[i] 和最长末尾下降状态长度 down[i], 结合图片可知，down[i] 一定从 up[i] 转移过来，up [i] 一定从 down [i] 转移过来。

通过`删除 == 忽略`的法则：如果 nums[i - 1] < nums[i] 也就是处于上升状态时，我们只对 up 进行更新，而对 down 来说，不必要进行更新。 nums[i - 1] > nums[i] 时同理，只对 down 进行更新，up 不用管。

通过这张图，我们可以直接写出对应代码：

```c++
int wiggleMaxLength(vector<int>& nums) {
    int n = nums.size();
    if (n < 2) {
        return n;
    }
    vector<int> up(n), down(n);
    up[0] = down[0] = 1;
    for (int i = 1; i < n; i++) {
        if (nums[i] > nums[i - 1]) {
            up[i] = down[i - 1] + 1;
            down[i] = down[i - 1];
        } else if (nums[i] < nums[i - 1]) {
            up[i] = up[i - 1];
            down[i] = up[i - 1] + 1;
        } else {
            up[i] = up[i - 1];
            down[i] = down[i - 1];
        }
    }
    return max(up[n - 1], down[n - 1]);
}

// 空间优化代码：
int wiggleMaxLength(vector<int>& nums) {
        int up = 1, down = 1, n = nums.size();
        for (int i = 1; i < n; i ++ ) 
            if (nums[i] > nums[i - 1]) up = down + 1;
            else if (nums[i] < nums[i - 1]) down = up + 1;
        return max(up, down);
    }
```



## 324. 摆动排序 II 
题意：将给定的 `nums[] 数组` 重新排序后，变为一个摆动数组，且规定`初始时`：第一个元素 < 第二个元素

模拟 + 排序 + 贪心

前导知识，模拟分割：模拟分割一个数组分割成数量相同，或者只相差 1 个的前后两部分：

1 > 前半部分元素数量 <= 后半部分：
```c++
vector<int> t1, t2;
for (int i = 0; i < n; i ++ )
    if (i <= n / 2 - 1) t1.push_back(nums[i]);
    else t2.push_back(nums[i]);
```

2 > 前半部分元素数量 t1.size() >= 后半部分 t2.size()
```c++
vector<int> t1, t2;
for (int i = 0; i < n; i ++ )
    if (i <= ceil(n / 2.0) - 1) t1.push_back(nums[i]);
    else t2.push_back(nums[i]);
```

一、数学思想：

我们将 nums[] 进行从小到大排序，然后分割为前后两部分： t1 , t2 依次穿插组合成一个新数组。但是会出现错误：因为可能有相同元素穿插到了一起，导致不是严格的摆动序列，例：[1, 2, 2] 与 [2, 3, 3], 该如何避免这种情况？我们将 t1 与 t2 都进行翻转，然后再进行穿插，这就避免了在原数组中间部分可能导致的相同元素遇到一起。也就是让相同的元素隔开，隔得越远越好，例：[2, 2, 1] 与 [3, 3, 2] 这就使得前部分 t1 中的 2 和后部分 t2 中的 2 相隔很远了。同时，我们还要保证 t1.size() >= t2.size(), 否则可能导致最后是`连续的两个 t2 内的相同元素相邻地`加入到结果中。也会导致出错：例：[2 1] 与 [4 3 3] 

二、代码实现：代码实现和数学思想类似，没有巧妙的转换

```c++
int n = nums.size();
vector<int> t1, t2;
sort(nums.begin(), nums.end());
for (int i = 0; i < n; i ++ )
    if (i <= ceil(n / 2.0) - 1) t1.push_back(nums[i]);  // 保证 t1.size() >= t2.size()
    else t2.push_back(nums[i]);
reverse(t1.begin(), t1.end()); reverse(t2.begin(), t2.end());
for (int i = 0, j = 0, k = 0; i < n; ) {
    nums[i ++ ] = t1[j ++ ];
    if (k < t2.size()) nums[i ++ ] = t2[k ++ ];
}
return; 

```

## LeetCode 354. 俄罗斯套娃信封问题

直接描述数学思想：

我们将信封宽度作为第一优先级进行从小到大依次排序，高度作为`第二优先级`从大到小进行排序。第一个优先级排序我们能理解，就是为了按顺序拿取信封，但为什么要进行第二个优先级的排序呢？这是因为为了减少代码时间复杂度，因为当宽度相同时，我们无论如何都不能将信封塞进另一个信封，所以为了避免`在高度上`进行最长递增子序列搜索时`还要去判断宽度`是否相同，我们直接将高度进行从大到小排序，导致相同宽度的信封时在高度这一维度上不可能出现有递增子序列长度大于 1 的情况(也就是在高度维度上完全递减)。

代码：
```c++
int f[100100];
f[1] = 1;
int n = envelopes.size();
sort(envelopes.begin(), envelopes.end(), cmp);
int ans = 1;
for (int i = 2 ; i <= n ; i ++ ) {
    f[i] = 1;
    for (int j = 1; j < i; j ++ ) 
        if (envelopes[j - 1][1] < envelopes[i - 1][1])
            f[i] = max(f[i], f[j] + 1);
    ans = max(ans, f[i]);
}
return ans;
```

利用二分 + 贪心来写最长递增子序列问题(该解法在算法笔记中详细分析过了，现在直接用)：
```c++
bool cmp(const vector<int> &a, const vector<int> &b){
    if (a[0] != b[0]) return a[0] < b[0];
    else return a[1] > b[1];
}
int n = envelopes.size();
sort(envelopes.begin(), envelopes.end(), cmp);
vector<int> g;
for (auto e : envelopes) {
    if (g.empty()) g.push_back(e[1]);
    int l = 0, r = g.size() - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (g[mid] >= e[1]) r = mid;
        else l = mid + 1;
    }
    if (g[l] < e[1]) g.push_back(e[1]);
    else g[l] = e[1];
}
return g.size();

```
## LeetCode 375. 猜数字大小 II
题意：在我们按照题目限定的猜法(即若猜大则下一次猜小，若猜小则下一次猜大)来猜答案的情况下，我们最多花费多少钱能确保我们以任何符合题意的方案来猜答案都可以保证我们能在钱没有花光的情况下猜对答案赢得游戏。

依照这题的难度来说，直接背算法和答案：区间DP
定义 f(l, r) 在范围 (i, j) 内确保胜利的最少金额，目标是计算 f(1,n)。

那么我们去`思考子问题`：一定是去寻找一个分割点 k 使得 [l, r] 分割为 [l, k - 1], k, [k + 1, j] 此时我们就得到子问题：[l, k - 1] 与 [k + 1, j]。

那么我们的 [l, r] 的花费一定是这两个子问题之中的 `最大花费子问题` + k。只有这样才能算出我们至少能够得到答案，否则低于这个花费可能猜不到正确数字，即钱不够。

而我们一定要去取这些分割情况的最小值，即`我们拥有能猜到正确数字的钱`之后，我们去考虑这样的所有分割的最小值。也就是最小化区间 [l, r] 的花费。那么逐渐进行递归即可。此题用记忆化搜索更加爽。


可以看到以下两种方法进行遍历时是不同的：
```c++
// 方法 1 按照区间从小到大，从左到右遍历：
class Solution {
public:
    int getMoneyAmount(int n) {
        int f[300][300];
        for (int i = n - 1; i >= 1; i--) 
            for (int j = i + 1; j <= n; j++) {
                f[i][j] = j + f[i][j - 1];
                for (int k = i; k < j; k++) {
                    f[i][j] = min(f[i][j], k + max(f[i][k - 1], f[k + 1][j]));
                }
            }
        return f[1][n];
    }
};

// 方法二 ： 按照区间长度，依次枚举左端点
int f[300][300];
memset(f, 0, sizeof f);
for (int len = 2; len <= n; len++) 
    for (int l = 1; l + len - 1 <= n; l++) {
        int r = l + len - 1;
        int &res = f[l][r] = 0x3f3f3f3f;
        for (int x = l + 1; x <= r; x++) 
            res = min(res, max(f[l][x - 1], f[x + 1][r]) + x);
    }

return f[1][n];
```
## LeetCode 368. 最大整除子集 这题是 排序 + 数学 + 动态规划，如果不利用数学优化，则时间复杂度为 O(n^3), 利用数学优化则为O(n^2)
数学特性：
1. 如果整数 a 是整除子集 S 的最小整数 b 的约数(即 a | b )，那么可以将 a 添加到 S 中得到一个更大的整除子集;

2. 如果整数 c 是整除子集 S 的最大整数 d 的倍数(即 d | c )，那么可以将 c 添加到 S 中得到一个更大的整除子集。

我们只需要第二个结论，就可以完成对他的动态规划： 先对元素进行排序，然后令 f(i) 以 i 为结尾元素对应的最大整除数组方案

我们只要考虑最后一个最大的数字即可


这里的`数学结论`就像是：我们可以用 `图的连接边 + DFS` 来表示对除数的连接：(a / b) * (b / c) = a / c。这样的数学很关键。

```c++
vector<int> largestDivisibleSubset(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int> > f(n);
    f[0] = {nums[0]};
    int maxSize = 1;
    vector<int> res = f[0];
    for (int i = 1; i < n; i ++ ) {
        f[i] = {nums[i]};
        for (int j = i - 1; j >= 0; j -- ) 
            if (nums[i] % nums[j] == 0) {
                vector<int> tmp = f[j];
                tmp.push_back(nums[i]);
                if (tmp.size() > f[i].size()) f[i] = tmp;       // 更新最大整除子集 f[i]
            }
        if (f[i].size() > maxSize) res = f[i], maxSize = f[i].size();
    }
    return res;
}
```


## 357. 统计各位数字都不同的数字个数
直接利用数学来接触这道题目，首先直接手算出 n = 0 和 n = 1 的情况

然后利用`排列` + `组合`数学知识：

当 n >= 2时：

`最高位只能为 1 ~ 9，剩下 n - 1 位从余下 9 位数中进行排列组合`

```c++
if (n == 0) return 1;
if (n == 1) return 10;
int cur = 9, ans = 10;
for (int i = 2; i <= n; i ++ ) 
    cur *= 11 - i;      // 计算 x 为 i 位时的不同数字个数，这个就是排列的计算公式，从 9 开始往前面乘
    ans += cur;         // 一直累加到 x 为 n 位时的不同数字个数
}
return ans;
```


## 396. 旋转函数
和那个 PAT 的分数计算题目相似，都是数学预处理类的题目，通过我们数学手动预处理，我们得到了以下公式：
f(i) = f(i - 1) + sum - n * nums[n - i];

直接通过这个公式写出代码即可：

```c++
int sum = 0, n = nums.size(); 
for (auto e : nums) sum += e;
int f[100100]; 
memset(f, 0, sizeof f);
for (int i = 0; i < n; i ++ ) f[0] += i * nums[i];
int ans = f[0];
for (int i = 1; i < n; i ++ ) {
    f[i] = f[i - 1] + sum - n * nums[n - i];
    ans = max(f[i], ans);
}
    
return ans;
```
## 397. 整数替换
直接按照题目意思写出 DFS 函数即可，唯一值得注意的是，这里只能用 unordered_map 来进行记忆化搜索，不能用到数组，因为数组没有那么大。

直接根据题目写代码：这里没有记忆化搜索，可以直接过
```c++
class Solution {
public:
    typedef long long LL;
    int DFS(LL n) {
        LL ans;
        if (n == 1) return 0;
        if (n % 2) ans = min(DFS(n + 1), DFS(n - 1)) + 1; 
        else ans = DFS(n / 2) + 1;
        return (int)ans;
    }
    int integerReplacement(int n) {
        return DFS(n);
    }
};

```

## 403. 青蛙过河
使用动态规划的方法，令f(i, k) 表示青蛙能否达到「现在所处的石子编号」为 i 且「上一次跳跃距离」为k的状态。

f(i, k) = f(j, k - 1) | f(j, k) | f(j, k + 1)

式中 j 代表了青蛙的「上一次所在的石子编号」从 `0 ~ i - 1`，满足 stones[i] - stones[j] = k,


代码细节：j 从后往前枚举，当上一次跳跃距离 k 超过 j + 1 时，我们可以停止跳跃：编号为 0 的时候我们最多只能跳 1 个距离，且每跳一次只能`将最远跳跃距离 + 1`，所以：`编号为 x 时，我们最多只能跳出 x + 1 的距离`，所以：在第 j 个石子上我们最多只能跳出 j + 1 的距离，
```c++
int n = stones.size();
int f[2010][2010];
memset(f, 0, sizeof f);
f[0][0] = 1;
for (int i = 1; i < n; i ++ ) 
    for (int j = i - 1; j >= 0; j -- ) {
        int k = stones[i] - stones[j];
        if (k > j + 1) break;       // 如果距离太大则一定不能从 j 这个编号的石子跳过来
        f[i][k] = f[j][k - 1] || f[j][k] || f[j][k + 1];        // 这里一定要用 || ，不能用 | ，不知道为何？
        if (i == n - 1 && f[i][k]) 
            return 1;
    }
return 0;
```

## 410. 分割数组的最大值  -> 前缀和 + 动态规划 或  二分 + 贪心 
动态规划写法，状态设计：设集合 (i, m) 为：前 0...i 个元素，分割成 m 个连续子数组段的所有方案，f(i, m) 为这些方案的最优解。我们通过枚举 (i, m) `最后一段的情况`来划分为`子集 (j, m - 1)` + `最后 [i..j] 这一段` ,由此写出状态转移方程，并用代码实现这个状态转移方程。`利用前缀和来在 O(1) 的时间复杂度实现计算最后一段的和`

```c++
memset(f, 0x3f, sizeof f);
f[0][0] = 0; S[0] = 0;
for (int i = 1; i <= n; i ++ )
    S[i] = S[i - 1] + nums[i - 1];
// 这个状态转移函数的代码实现是非常有难度的：
for (int i = 1; i <= n; i ++ )
    for (int j = 1; j <= min(i, m); j ++ )  //这里是体现了 i 最多划分为 i 段或者题目要求的 m 段
        for (int k = j - 1; k < i; k ++ )   // 这里体现了最后被划分的段为 [k + 1..i]
            f[i][j] = min(f[i][j], max(f[k][j - 1], S[i] - S[k]));
        
```


二分 + 贪心写法：   
```c++
vector<int> a;
int n;
bool check(int x, int m){   // 贪心检查是否满足题目分割条件
    int sum = 0, cnt = 1;
    for (int i = 0; i < n; i ++ ) {
        if (sum + a[i] > x) { cnt ++ ; sum = a[i]; }
        else sum += a[i] ;
    }
    return cnt <= m;        
}
int splitArray(vector<int>& nums, int m) {
    a = nums;
    n = a.size();
    int l = 0, r = 0;
    for (int i = 0; i < n; i ++ ) {
        r += a[i];          // 初始 r 边界为数组元素总和 sum
        l = max(l, a[i]);   // 初始 l 边界为数组中最大元素值
    }

    while (l < r) {     // 二分搜索最优解，check 满足：mid (每段子数组能够 "被允许" 的最大和) 越大，越能符合限定条件
        int mid = l + r >> 1;
        if (check(mid, m)) r = mid;
        else l = mid + 1;
    }

    return l;
}

```


## 413. 等差数列划分 -> 子数组

数学描述：1.等差数列是长度为 3 才算：

如果以 i 为结尾的元素能够加入前一个子数组等差数列(即两者的 dif 相同)，那么答案就会增加 len(len为算上当前的元素和前面的等差子数组一起组成的等差子数组的长度) - 2 个等差数列，例：[1,2,3] -> [1,2,3,4] 则增加了 4 - 2 个，分别为：[2,3,4] 和 [1,2,3,4]。其他例子同理：[1,2,3,4] -> [1,2,3,4,5] 则增加了 5 - 2 = 3 个，分别为：[3,4,5], [2,3,4,5], [1,2,3,4,5]。

代码实现，在代码实现上，我们不需要想数学思想那样维护一个当前等差子数组长度，我们只需要维护一个 `增加变量 t ` 即可，因为每次如果能增加的话，`当前等差子数组长度`一定是线性增长的，所以 t 也是线性增长的：t = 1, 2, 3, 4, 5, ....  然后每次让 ans += t 即可

代码：
```c++
int numberOfArithmeticSlices(vector<int>& nums) {
    int n = nums.size();
    if (n < 3) return 0;
    int f[n]; memset(f, 0, sizeof f);
    int dif = nums[1] - nums[0], t = 1, ans = 0; 
    for (int i = 2; i < n; i ++ ) {
        if (nums[i] - nums[i - 1] == dif) ans += t, t ++ ;
        else dif = nums[i] - nums[i - 1], t = 1;
    }
    return ans;
}
```
上述为等差数列的`子串`版本，字串版本一定是 以 i 为结尾，连续的，所以我们才能这样写，`有点滑动窗口的感觉`

我们来看各种版本的等差数列：
1. `1027. 最长等差数列` 这题需要用到 ofst 来做偏移量，由于值域被限制得很小，那么他的差值的绝对值也被限制在一定的范围，那么我们可以用 f[i][dif] 来表示：以 i 为结尾，dif 为公差的所有的子序列里，最长的子序列的长度。以此来做状态转移方程
2. `1218. 最长定差子序列` 这题同样是子序列版本的题目，但是是定差子序列，那么我们只要枚举合法性子序列即可。即只能是 f[i] - f[j] == dif 这种才能合法转移，否则忽略其转移。


## 446. 等差数列划分 II -> 子序列
定义： 我们首先考虑`至少有两个`元素的等差子序列，下文将其称作弱等差子序列。

由于尾项和公差可以确定一个等差数列，因此我们定义状态集合 (i, d) 表示尾项为 nums[i]，公差为 d 的所有弱等差子序列的方案。f(i, d) 为这样规定下的方案个数

注：第二个维度 d 范围太大，只能用哈希表来实现集合 (i, d)。这个集合非常巧妙，包含了比较奇妙的语法知识：

```c++
typedef long long LL;

int ans = 0, n = nums.size();
unordered_map<LL, int> f[1010];     // 注意这个东西可以组成一个 f[1010][LL] 的数组，非常奇妙
// 这里对数学的代码实现也设计的非常巧妙，注意学习：
for (int i = 0; i < n; i ++ )
    for (int j = 0; j < i; j ++ ) {
        LL d = 1LL * nums[i] - nums[j];
        int cnt = f[j].count(d) ? f[j][d] : 0;      
        ans += cnt;     // 注意，这里第一次出现只含 2 个元素的等差子序列时，cnt = 0，即 ans += 0
        f[i][d] += cnt + 1; // 这个就是第一次是只有 2 个元素，则添加进去 1 ，然后第二次搜到相等的 d 时，那么就可以让 ans += 1, 然后 f[i][d] += 1 + 1,即让下次的这个数列可以增加 2
    }

```

## 435. 无重叠区间  -> 排序 + 贪心
这题就是选会议问题转换一下，这里直接上代码：

```c++
bool cmp(const vector<int> &a, const vector<int> &b){
    return a[1] < b[1];
}

sort(itv.begin(), itv.end(), cmp);
int ans = 1, curR = itv[0][1], n = itv.size();
for (int i = 1; i < n; i ++ ) {
    if (itv[i][0] >= curR) { ans ++ ; curR = itv[i][1]; }
    else continue;
}
return n - ans;
```

## 464. 我能赢吗
这个关键的点在于：双方不能选取已经选过的数字，所以我们用 state 来表示数字 1 ~ n ，其中数字 1 在第 0 位，为第 0 个元素。

位运算技巧：

判断第 i 个元素是否被用过：((state >> i) & 1) == 1  `一定要有这个 & ，否则像 1100， 1000 这种都会判对`

state 添加第 i 个元素 ： state | (1 << i)


为什么只需要去记忆 state 而不需要去记忆 cur ？

答：因为实际上返回值是与 cur 无关的，你可以将 cur 看作是一个全局的回溯变量，即：如果我们将 cur 参数去除的话，可以这样写：
1. cir += i;
2. DFS(state | (1 << i));
3. cur -= i;

```c++
int memo[1 << 22];
int tar, n;
bool DFS(int state, int cur) {
    if (memo[state] != -1) return memo[state];
    int &res = memo[state]; res = 0;
    for (int i = 1; i <= n; i ++ ) {
        if (((state >> i) & 1) == 1) continue;
        if (i + cur >= tar) res = 1;
        res |= !DFS(state | (1 << i), cur + i);     // 对手失败我就能赢
        if (res) return res;
    }
    return res;
}
bool canIWin(int _n, int _tar) {
    memset(memo, -1, sizeof memo);
    n = _n, tar = _tar;
    if ((1 + n) * n / 2 < tar) return 0;
    return DFS(0, 0);
}

// 解答为何不需要 cur 作为 memo 记忆化的参数：
int memo[1 << 22];
int tar, n, cur;
bool DFS(int state) {
    if (memo[state] != -1) return memo[state];
    int &res = memo[state]; res = 0;
    for (int i = 1; i <= n; i ++ ) {
        if (((state >> i) & 1) == 1) continue;
        if (i + cur >= tar) res = 1;
        cur += i;
        res |= !DFS(state | (1 << i));
        cur -= i;
        if (res) return res;
    }
    return res;
}
bool canIWin(int _n, int _tar) {
    memset(memo, -1, sizeof memo);
    n = _n, tar = _tar;
    if ((1 + n) * n / 2 < tar) return 0;
    cur = 0;
    return DFS(0);
}
```

## 365. 水壶问题
经典的 BFS 问题，使用状态搜索即可，也就是`抽象化为 最短路 `，来搜索到达目标状态的最短路。

```c++

int v1, v2;
void change(int &x, int &y, int mode){
    if (mode == 0)  x = v1;
    else if (mode == 1) y = v2;
    else if (mode == 2) x = 0;
    else if (mode == 3) y = 0;
    else if (mode == 4)  // x -> y
        if (x + y >= v2) x = x - (v2 - y), y = v2;
        else y = x + y, x = 0;    
    else if (mode == 5)   // y-> x
        if (x + y >= v1) y = y - (v1 - x), x = v1;
        else x = x + y, y = 0;
}
bool canMeasureWater(int a, int b, int tar) {
    auto fun = [] (const PII &p) { return hash<int>()(p.first) ^ hash<int>()(p.second << 1); }; 
    unordered_set<PII, decltype(fun)> S;

    v1 = a, v2 = b;
    PII original = {0, 0};
    queue<PII> q;
    q.push(original);
    while (q.size()) {
        auto t = q.front(); q.pop();
        if (S.count(t)) continue;
        else S.insert(t);
        auto [x, y] = t;
        if (x + y == tar || x == tar || y == tar) return 1;
        for (int i = 0; i < 6; i ++ ) {
            int nex = x, ney = y;
            change(nex, ney, i);
            q.push({nex, ney});
        }
    }
    return 0;
}


```
## 466. 统计重复个数
题意：

给你两个循环字符串 str1, str2。其中 str 的循环次数为 n1, 单位字符串为 s1； str2 的循环次数为 n2, 单位字符串为 s2。例：

s1 为 "abc", n1 为 5，则循环字符串 str1 为：`abcabcabcabcabc`

要求将 str2 循环 m 次使得 [str2, m] 为 str1 的`子序列`，求 m 最大为多少

## 467. 环绕字符串中唯一的子字符串      字符集 + dp
`关键`：对于每个重复的字串来说，我们是`不能`将他`重复计入答案`中的，必须要`去重`。所以有了这个关键的点，我们才能懂得该如何去做：这就`类似于一个求公差固定为 1 的最长等差数列问题了`。

题意：统计 s 中的 `不同非空子串个数` s.t. 这些子串都是 base 的子串， 其中 `base 是一个无限循环字符串`，从 ...abc... -> ...wxyzabc... 。

关键分析：这个题目如果分析到了这一点，说明已经很理解题意，并能写出答案了：

从 s 中选出：以任意字符 c 为结尾的符合题意的，且是最长的一个子串。如果有相同的 c 出现，我们一定选取符合前面所说的最优的那个 c 。

那么此时我们累计 mp[c] 中所有的子串长度即可。就如同前面的 `413. 等差数列划分` 类似，每次增加一个末尾的字符，那么就会 `增加 len 个不同的子串` ，例：

"abcd" -> "abcde" 那么后者比前者多了 5 个不同的子串，方法是以 e 为结尾开始往前枚举。则转移方程为：`f[i] = f[i - 1] + 1;`

这题的本质是：公差固定的等差子串。我建议与 `413. 等差数列划分(子串)` 和 `1218. 最长定差子序列` 进行类比，因为它就是这两个的结合体。
```c++
unordered_map<char, int> mp;
int findSubstringInWraproundString(string s) {
    int n = s.size(); mp[s[0]] = 1;
    int f[n]; memset(f, 0, sizeof f); f[0] = 1;
    for (int i = 1; i < n; i ++ ) {
        if ((s[i] - s[i - 1] + 26) % 26 == 1)       // `定差` 的判断条件
            f[i] = f[i - 1] + 1;                    // 子串的转移方程
        else   
            f[i] = 1;                               // 子串的转移方程
        mp[s[i]] = max(mp[s[i]], f[i]);
    }
    int ans = 0;
    for (auto [k, v] : mp)                      // 定结尾型滑动窗口的个数
        ans += v;
    return ans;
}
```

## 698. 划分为k个相等的子集 & 473. 火柴拼正方形  ->  一种类型的题
注意这种`符合某一条件才能跳入下一条件`的状态搜索顺序，是和经典的 `八皇后问题` 的模式写法相同的，而`小猫爬山`则是另一种相类似但是却不同的写法。

***区分如下：***

八皇后问题：只能选取`某限制条件下`的答案

小猫爬山：在`某一限制条件下`，选取`最优解`

此题和 `473. 火柴拼正方形` 一模一样，需要利用到状态压缩 + 记忆化搜索。

注意：

1 > 判断是否使用过某一元素的代码，注意这个左移右移一定要分清楚： `if (((state >> i) & 1) == 1)`

2 > 加入某一元素：`state |= 1 << i;`

3 > 删除某一元素：`state ^= 1 << i;  ( i 一定在 state 中)` 或者 `state &= ~(1 << i); ( i 可以不在state中)`

## 474. 一和零
这道题和 01 背包非常类似，但是`值得注意的不同点`是：
根据实际问题来说，01背包问题不会出现`体积为 0 价值非 0 的物品`但是这里会出现 num0 或者 num1 为 0 (相当于某一个维度的体积为 0 )的情况。但是我犯了一个错误，就是我从体积 1 开始遍历了，导致有用例通不过。所以还是得从 0 开始。但是如果使用空间优化的 01 背包做法就不会出现这种问题了

```c++

int count0(string s) {
    int ans = 0;
    for (auto e : s) if (e == '0') ans ++ ;
    return ans;
}

int findMaxForm(vector<string>& strs, int m, int n) {
    memset(f, 0, sizeof f);
    for (int i = 1; i <= strs.size(); i ++ ) {
        int num0 = count0(strs[i - 1]);
        int num1 = strs[i - 1].size() - num0;
        for (int j = m; j >= num0; j -- )
            for (int k = n; k >= num1; k -- ) 
                f[j][k] = max(f[j][k], f[j - num0][k - num1] + 1);

    }
    return f[m][n];
}

```

## 

## 514. 自由之路    ->     阶段性动态规划，类似于 01 背包这种阶段性思想。
这道题处理循环字符的 `位置之差` 计算技巧和 `467. 环绕字符串中唯一的子字符串` 类似，但不完全相同

令 `f[i][j]` 为：从前往后，拼写出 key 的第 i 个字符， ring 的第 j 个字符的最少步数。(`即此时 key[i] == ring[j] 且 ring[j] 与 12 点钟方向对齐`)

直接上算法步骤:

1 > 由于我们转动的是 ring，所以我们要`记录下 ring 中所有字符的下标位置`，以每个不同的小写字母为一组，由于可以有重复的小写字母，所以不同的字母对应的一组中含有多个不同的`下标位置元素`，以便于  ->  `让 ring 的位置下标做差来**模拟**转动的次数`

2 > 初始化 f[0] 即 key 的首个字符与 ring 中`相同字符`能够对应拼写上的最少步数

3 > 利用阶段性状态转移思想，即当前状态是由上一阶段转移过来的：`f[i][j] = Transfer(self, f[i - 1][pre_pos])`


》当数组或字符串为循环数组或循环字符串，且`循环节长度为 len 时`，`计算位置之差`的两种方法：
1. `dif = min((pos1 - pos2 + len) % len, (pos2 - pos1 + len) % len)`
2. `dif = min(abs(pos1 - pos2), n - abs(pos1 - pos2))`

代码：

``` c++
int n = ring.size(), m = key.size();
vector<int> pos[26];
for (int i = 0; i < n; i ++ ) pos[ring[i] - 'a'].push_back(i);  // 记录每个 ring 中 的字符的下标: **为了模拟需要转动的次数**

int f[110][110];
memset(f, 0x3f, sizeof f);

for (auto e : pos[key[0] - 'a']) f[0][e] = min(e - 0, (0 - e + n) % n) + 1; // 初始化 f[0][...]，我们只在相同的字符中去选取位置进行模拟旋转

for (int i = 1; i < m; i ++ )   // 遍历 key[i]
    for (auto j : pos[key[i] - 'a'])    // 这一步代码理解：我们只在 key[i] 和 ring[j] 相等时进行模拟旋转次数，如果不等则没必要进行模拟旋转
        for (auto k : pos[key[i - 1] - 'a'])
            f[i][j] = min(f[i][j], f[i - 1][k] + min((j - k + n) % n, (k - j + n) % n) + 1);
int ans = INF;
for (int j = 0; j < n; j ++ )
    ans = min(ans, f[m - 1][j]);
return ans;
```

## 486. 预测赢家
应该用该题与`464. 我能赢吗`做对比，这题由于题目要求是在区间两端进行选择，所以可以利用 `区间 DP` 而没必要利用状态压缩。这就是此题与`464. 我能赢吗`的最大区别之一。

这里需要注意的是，当轮到我选择时，对于任意一个 `f[i][j]` 其后选 sec 是受先选 fir 的约数，即 sec 是由 `f[i][j].fir` 是选 left 还是选 right 决定的，而我之前没考虑这个原因，直接让：

`f[i][j].fir = max(left, right), f[i][j].sec = max(f[i + 1][j].fir, f[i][j - 1].fir)` 这是错误的转移方程。它的前半部分没有问题，但是后半部分没有考虑到 fir 对 left 和 right 选择会对 sec 的选择产生限制，导致 sec 在无论 fir 选择 right 还是 left 都去选一个有利于 sec 的最优解，这是错误的，举例： [1, 5] 

如果按正常题目要求，那么 sec 一定是 1，但是如果按错误的转移方程，那么 sec 就是 5，因为它没有考虑当前的 fir 的选取情况而是直接去选一个最好的东西


```c++
struct Node {
    int fir, sec;
};
bool predictTheWinner(vector<int>& nums) {
    int n = nums.size();
    Node f[22][22];
    for (int i = 1; i <= n; i ++ )
        f[i][i].fir = nums[i - 1], f[i][i].sec = 0;
    
    for (int i = n - 1; i >= 1; i -- )
        for (int j = i + 1; j <= n; j ++ ) {
            int left = nums[i - 1] + f[i + 1][j].sec;
            int right = nums[j - 1] + f[i][j - 1].sec;
            if (left > right) 
                f[i][j].fir = left, f[i][j].sec = f[i + 1][j].fir;
            else 
                f[i][j].fir = right, f[i][j].sec = f[i][j - 1].fir;
        }
    
    if (f[1][n].fir >= f[1][n].sec) return 1;
    else return 0;    
}
```

### 进阶：
如何去进一步地写出更好地状态表达：
定义`f(i, j)`表示当数组剩下的部分为下标 i 到下标 j 时，即在下标范围`[i, j]`中，`当前玩家与另一个玩家的分数之差的最大值`，注意当前玩家不一定是先手。而是正在取数的当前回合玩家。

那么 `f(i, j) = max{nums_i - f(i + 1, j), nums_j - f(i, j - 1)}` 意思是：当前玩家与另一个玩家的分数之差分为两种情况：

1 > 选择 `nums_i` 则，通过数学分析有以下等式成立：
    `f(i, j) = player1_grade - player2_grade = nums_i - (player2_grade - (player1_grade - nums_i)) = nums_i - f(i + 1, j)`
    分析：其中 `player2_grade - (player1_grade - nums_i)` 就是下一个当前玩家(player2) 所能够获得的最大分数之差，因为player1已经选择过了nums_i, 所以这个 `player1_grade - nums_i` 是在 `[i + 1, j]` 这个区间中 player1 所能获得的最优解分数。所以 `f(i + 1, j)` 代表了 `player2_grade - (player1_grade - nums_i)`。

2 > 同理，当选择 `nums_j` 时，同样这样分析即可。

直接上代码：
```c++
int f[22][22], n = nums.size();
memset(f, 0, sizeof f);
for (int i = 1; i <= n; i ++ ) f[i][i] = nums[i - 1];
for (int len = 2; len <= n; len ++ )
    for (int l = 1; l <= n - 1 && l + len - 1 <= n; l ++ ) {
        int r = l + len - 1;
        f[l][r] = max(nums[l - 1] - f[l + 1][r], nums[r - 1] - f[l][r - 1]);
    }
return f[1][n] >= 0;
```

换个方式来遍历，使得我们可以节省一维空间消耗：

```c++
int f[22], n = nums.size();
memset(f, 0, sizeof f);
// 我们来详细分析一下这个空间压缩的过程：实际上就是隐藏了一个维度而已：
// 所以当外层循环到 l，内层循环到 r 时，f[r] 就代表了 f[l][r] ，只是隐藏了一个维度而已  
for (int i = 1; i <= n; i ++ ) f[i] = nums[i - 1];  
for (int l = n - 1; l >= 1; l -- )      
    for (int r = l + 1; r <= n; r ++ )
        f[r] = max(nums[l - 1] - f[r], nums[r - 1] - f[r - 1]);
return f[1][n] >= 0;
```



## 494. 目标和    ->     数学 + 01背包
数学转化，设全部元素的和为 sum，前面添加负号的元素总和为 neg(negative) ，前面添加正号的元素总和为 pos(positive)，则有：

1. neg + pos = sum
2. pos - neg = tar

由上述两个二元方程组可解出：`pos = (sum + tar) / 2`, `neg = (sum - tar) / 2`

此时我们可以利用 01背包来计算方案(`组合非排列类型`)数目：

举例利用 neg 作为背包容量来写：
```c++
const int N = 1e6 + 10;
int f[N];
int findTargetSumWays(vector<int>& nums, int target) {
    memset(f, 0, sizeof f); f[0] = 1;
    int neg, sum = 0, n = nums.size();
    for (auto e : nums) sum += e;
    if ((sum - target) % 2 || sum - target < 0) return 0;  // 如果二元方程组无解，则返回 0
    neg = (sum - target) / 2;
    for (int i = 0; i < n; i ++ )
        for (int j = neg; j >= nums[i]; j -- )
            f[j] += f[j - nums[i]];
    return f[neg];
}
```

## 516. 最长回文子序列   -> 序列DP 
此题可以与最长回文子串来比较，其中，分别是子串，子列的DP，我们可以通过转移方程来观察它们两个的不同之处，直接上代码：


```c++
// 子串：
memset(f, 0, sizeof f);
for (int i = 1; i <= n; i ++ ) f[i][i] = 1;

for (int len = 2; len <= n; len ++ )
    for (int l = 1; l <= n && l + len - 1 <= n; l ++ ) {
        int r = l + len - 1;
        f[l][r] = (s[l] == s[r] && (f[l + 1][r - 1] || l + 1 >= r - 1)) ? f[l + 1][r - 1] + 2 : 0;
    }

// 子序列：
memset(f, 0, sizeof f);
for (int i = 1 ;i  <= n; i ++ ) f[i][i] = 1;

int f[1010][1010], n = s.size();
memset(f, 0, sizeof f);
for (int i = 1 ;i  <= n; i ++ ) f[i][i] = 1;
int ans = 1;
for (int len = 2; len <= n; len ++ )
    for (int l = 1; l <= n && l + len - 1 <= n; l ++ ) {
        int r = l + len - 1;
        f[l][r] = s[l - 1] == s[r - 1] ? f[l + 1][r - 1] + 2 : max(f[l + 1][r], f[l][r - 1]);   // 这里的判断条件与子串不一样，还有后面的转移方程也是不一样的。
        ans = max(ans, f[l][r]);
    }
return ans;

```


## 518. 零钱兑换 II
凑出体积为 j 的方案数的一般写法： `f[j] += f[j - nums[i]]`，这里的题意要求是固定了只能组合不能排列

```c++

int f[5010];
memset(f, 0, sizeof f); f[0] = 1;
int n = nums.size();
for (int i = 0; i < n; i ++ ) {
    for (int j = nums[i]; j <= tar; j ++ )
        f[j] += f[j - nums[i]];
}
return f[tar];
```

## 526. 优美的排列
我认为这是一道 `状态压缩 + 回溯参数辅助` 类型的题，就如 `我能赢吗` 这题一样，我们可以不用 idx 这个参数放在函数中，因为它可以放在全局变量中，我们加入 `这个参数` 只是为了辅助判断 `此时回溯` 已经`进行到哪一步`了。

这是一道 DFS 的题，准确来说，这是一道排列类型的题，所以 DFS 时，我们应该考虑排列的顺序来搜索答案，并不会导致重复。写法如下:

注：此题就算不进行记忆化，只用标准的 `累计答案版回溯(void 作为 DFS 类型，ans ++ 版本) + 剪枝` ，时间复杂度上也能通过该题
```c++

int memo[1 << 17];
int n;
int DFS(int idx, int state) {
    if (state == (1 << n) - 1) return 1;
    if (memo[state] != -1) return memo[state];
    int &res = memo[state]; res = 0;
    for (int i = 0; i < n; i ++ ) {
        if (((state >> i) & 1) == 1) continue;
        if ((idx + 1) % (i + 1) == 0 || (i + 1) % (idx + 1) == 0) {
            res += DFS(idx + 1, state | (1 << i));
        }
    }
    return res;
}
int countArrangement(int _n) {
    n = _n;
    memset(memo, -1, sizeof memo);
    return DFS(0, 0);
}

class Solution {
public:
    int state = 0, ans = 0, n;                 // state 可以用 vis[] 数组代替
    void DFS(int idx){                         // 标准的 void ，作为回溯
        if (idx == n) { ans ++ ; return; }     // ans ++ 版本
        for (int i = 1; i <= n; i ++ ) {
            if ((idx + 1) % i && i % (idx + 1) || (state >> i) & 1 == 1) continue;
            state |= (1 << i);
            DFS(idx + 1);
            state &= ~(1 << i);
        }
    }
    int countArrangement(int _n) {
        n = _n;
        DFS(0);
        return ans;
    }
};
```

动态规划方法：
从记忆化搜索的角度去考虑动态规划的转移情况：因为 state 在DFS爆搜算法中不能代表顺序，那么我们对他进行一个重新定义，使他能够代表状态集合并进行状态转移：state 代表所有选中的数且符合题意的所有排列方案。

那么`f[state]`解释为：已经选中的元素集合为 state，并且满足题意的排列方案，那么转移方程为：在 f[state] 的方案下，去挑选没选择过的数，且满足题意的数，然后在`f[state]`的基础上去添加即可，对比 `518. 零钱兑换 II` 的转移方程，有类似之处


## 542. 01 矩阵

记忆化搜索做法：

在理论上来说：`542. 01 矩阵` 是`需要往回搜索`来确定一条最短路径的，因为可能一条路径的`最短路径`是`经过来时的点`而`产生的最优解`。所以在理论上来说，是需要往回搜索的。

但是在`使用深度优先搜索`时却不能使用`往回搜索`，除非你使用回溯算法去`搜索所有的路径`，而回溯算法的

那么我们如何解决超时问题呢？直接分四个搜索顺序来进行记忆化搜索：右下，右上，左下，左上。然后在这四种方案中找最小的值即可。注意这四个方向上的代码容易写错，在实现时必须按逻辑写成`右下，右上，左下，左上`，而不是写成`左，上，右，下`。数学优化：代码中只写左上的与右下的代码即可写出完整答案，正确性证明略。

[   [1,0,1,1,0,0,1,0,0,1],
    [0,1,1,0,1,0,1,0,1,1],
    [0,0,1,0,1,0,0,1,0,0],
    [1,0,1,0,1,1,1,1,1,1],
    [0,1,0,1,1,0,0,0,0,1],
    [0,0,1,0,1,1,1,0,1,0],
    [0,1,0,1,0,1,0,0,1,1],
    [1,0,0,0,1,1,1,1,0,1],
    [1,1,1,1,1,1,1,0,1,0],
    [1,1,1,1,0,1,0,0,1,1]]

更为直觉且更好用的方法(时间复杂度相同)：`超级源点 + SPFA`：构造 d[][] 数组，进行 SPFA 算法即可。
```c++
int m = mat.size(), n = mat[0].size();
vector<vector<int>> res(m, vector<int>(n));
vector<vector<int>> v(m, vector<int>(n));
queue<PII> q;

for (int i = 0; i < m; i ++ )
    for (int j = 0; j < n; j ++ )
        if (!mat[i][j]) { res[i][j] = 0; q.push({i, j}); v[i][j] = 1; }

while (q.size()) {
    auto [x, y] = q.front(); q.pop();
    int dis = res[x][y];
    for (int i = 0; i < 4; i ++ ) {
        int nex = x + dx[i], ney = y + dy[i];
        if (nex >= m || nex < 0 || ney >= n || ney < 0 || v[nex][ney]) continue;
        q.push({nex, ney}); res[nex][ney] = dis + 1; v[nex][ney] = 1;
    }
}
return res;
```


## 329. 矩阵中的最长递增路径

在理论上来说：
1. `542. 01 矩阵` 是`需要往回搜索`来确定一条最短路径的，因为可能一条路径的最短路径是经过来时的路径而产生的。所以在理论上来说，是需要往回搜索的。所以如果想用记忆化来解决超时问题，则要设计 4 种不同的搜索方案来解决该问题，并在最后`取这四种方案`的最小值：`ans[x][y] = min({res1[x][y], res2[x][y]...})`
2. `329. 矩阵中的最长递增路径` 是不需要往回搜索来确定一条最短路径的，因为往回搜索必然导致违反单调递增这一`合法`解。所以在理论上来说就不能往回搜索来处理`合法`答案。所以这题的代码可以直接用固定的搜索顺序 + 记忆化即可。

那么动态规划 + 记忆化搜索写法为：
```c++

int f[N][N];
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};
int m, n;
vector<vector<int> > maze;
void DFS(int x, int y){
    if (f[x][y] != -1) return;

    int &res = f[x][y]; res = 1;
    for (int i = 0; i < 4; i ++ ) {
        int nex = x + dx[i], ney = y + dy[i];
        if (!(nex >= 0 && nex < m && ney >= 0 && ney < n) || maze[x][y] >= maze[nex][ney]) continue;
        DFS(nex, ney);
        res = max(res, f[nex][ney] + 1);
    }
    
}
int longestIncreasingPath(vector<vector<int>>& matrix) {
    memset(f, -1, sizeof f);
    m = matrix.size(), n = matrix[0].size();
    maze = matrix;
    int ans = 0;
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ ) {
            DFS(i, j);
            ans = max(ans, f[i][j]);
        }
    return ans;

}
```

同`542. 01 矩阵`中，如果我们想从一个点往边上扩展来搜索，那么同样基本思想为`拓扑排序型BFS`,那么如何定义源点呢？按照人类的思想去想，肯定是将四周最小的点为源点去向四周去搜索，但是`这样想是错误的`!!!因为这是直观的感觉，而没有`深层`次地去理解`拓扑排序的抽象化定义`。

拓扑排序抽象化定义：拓扑排序一定是将边界条件作为源点。然后从边界开始，一步一步去往回搜索。就如同逆向思维一样，我们可以发现，为什么拓扑排序的源点是入度为 0 的点？因为这是一个边界条件，可能你们会说：边界条件不应该是出度为 0 的点吗？因为边界不是最终的条件吗？实际上对于抽象化拓扑排序来说，源点既可以是`入度为 0 的点`，也可以是`出度为 0 `的点，因为在`数学的抽象化概念`上来讲，这两者是等价的。只不过在我们人类解释时，前者为顺拓扑排序，后者为逆拓扑排序，但都符合拓扑排序的性质， reverse 一下序列，这两者就完全相等了。所以我们要从数学的抽象角度上去分析并解决问题，而不是总是使用直观的角度。

显然，这题`使用出度为 0 的点作为源点这样的定义`比入度为 0 作为源点的定义要更好。那么边界就是将比四周都大的点作为起点开始，`逆拓扑排序进行搜索`，直到搜到一个极小值点，此时不能再搜了。

```c++
int dirs[4][2] = { {0, 1}, {0, -1}, {1, 0}, {-1, 0} };
int longestIncreasingPath(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    vector<vector<int> > res(m, vector<int>(n)), outd(m, vector<int>(n)), v(m, vector<int>(n));
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ )
            for (int k = 0; k < 4; k ++ ) {
                int nei = dirs[k][0] + i, nej = dirs[k][1] + j;
                if (nei < 0 || nei >= m || nej < 0 || nej >= n) continue;
                if (mat[i][j] < mat[nei][nej]) outd[i][j] ++ ;
            }
    queue<PII> q;
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ )
            if (!outd[i][j]) { res[i][j]++; q.push({i, j});}
    int ans = 0;
    while (q.size()) {
        auto [x, y] = q.front(); q.pop();
        int &dis = res[x][y];
        ans = max(ans, dis);
        for (int i = 0; i < 4; i ++ ) {
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex < 0 || nex >= m || ney < 0 || ney >= n || res[nex][ney] || mat[nex][ney] >= mat[x][y]) continue;    // 注意这里一定要给出 `mat[nex][ney] >= mat[x][y]` 这个条件，因为要防止将同层(res相同)的节点入队和防止减去太多的outd，导致出错，比如防止一个入度为 0 的节点使得另一个入度为 0 的节点减为 -1.导致结果出错。
            if ( -- outd[nex][ney] == 0) {
                res[nex][ney] = dis + 1; 
                q.push({nex, ney});
            }
        }
    }
    return ans;
}

```

分层式拓扑排序模板：

```c++
int ans = 0, m = mat.size(), n = mat[0].size();
const int dirs[4][2] = { {0, 1}, {0, -1}, {1, 0}, {-1, 0} };
vector<vector<int> > outd(m, vector<int>(n));
for (int x = 0; x < m; x ++ )
    for (int y = 0; y < n; y ++ )
        for (int i = 0; i < 4; i ++ ){
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex >= m || nex < 0 || ney >= n || ney < 0) continue;
            if (mat[x][y] < mat[nex][ney]) outd[x][y] ++ ;
        }
queue<PII> q;
for (int x = 0; x < m; x ++ )
    for (int y = 0; y < n; y ++ )
        if (!outd[x][y]) q.push({x, y});
while (q.size()) {
    ans ++ ;
    int cnt = q.size();
    while (cnt -- ) { // 这里利用了分层式BFS，也就是每一次将一个层次的节点进行处理，这样可能更为清晰。
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 4; i ++ ) {
            int nex = dirs[i][0] + x, ney = dirs[i][1] + y;
            if (nex >= m || nex < 0 || ney >= n || ney < 0 || mat[nex][ney] >= mat[x][y]) continue;
            if (-- outd[nex][ney] == 0) q.push({nex, ney});
        }
    }
}
return ans;
```

## 553. 最优除法
这题的动态规划写法太难受了，但是有一个`代码模板实现数学的动态转移方程`值得学习：
从 `[l, r]` 中去枚举 `k in [l, r) // 左闭右开`, 那么就可以划分两个区间：`[l, k] 与 [k + 1, r]`
```c++
for (int len = 1; len <= n; len ++ )
    for (int l = 0; l < n && l + len - 1 < n; l ++ ) {
        int r = l + len - 1;
        for (int k = 0; k < r; k ++ ) {
            // process
            f[l][r] = Transfer(f[l][k], f[k + 1][r]);
        }
    }

for (int l = n - 1; l >= 0; l -- )
    for (int r = l; r < n; r ++ )
        for (int k = l; k < r; k ++ ) {
            // process
            f[l][r] = Transfer(f[l][k], f[k + 1][r]);
        }
```

## 576. 出界的路径数
这题`用记忆化搜索`的方法会比`动态规划`的转移方程代码实现要更易于理解。所以我们在这里介绍记忆化搜索的方法：

关键在于状态定义与边界定义：

状态定义：`memo[x][y][remain]` 代表从坐标 `(x, y)` 出发，还剩余 remain 步，能够出界的路径方案数目

1 > 如果 `(x, y)` 在边界外边则无论还剩下多少步，`memo[x][y][remain]` 都只能为 1。因为他已经出去了，无论剩下多少步都只有一条路径

2 > 如果 `(x, y) `在边界内，remain 为 0，则返回 0，因为他剩余步数已经为 0 了，但是还在界内，说明走不出去，方案只能为 0 

有了这两个边界条件，我们就可以直接写出代码了，状态转移方程在代码中体现：

```c++

int m, n;
LL memo[55][55][55];
const int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
LL DFS(int x, int y, int remain) {
    if (x < 0 || x >= m || y < 0 || y >= n) return 1;
    if (remain == 0) return 0;
    if (memo[x][y][remain] != -1) return memo[x][y][remain];
    LL &res = memo[x][y][remain]; res = 0;
    for (int i = 0; i < 4; i ++ ) {
        int nex = x + dirs[i][0], ney = y + dirs[i][1];
        res = (res + DFS(nex, ney, remain - 1)) % MOD;
    }
    return res;
}
int findPaths(int _m, int _n, int maxMove, int startRow, int startColumn) {
    memset(memo, -1, sizeof memo); m = _m, n = _n;
    return DFS(startRow, startColumn, maxMove);
}
```

## 583. 两个字符串的删除操作
这个可以和`516. 最长回文子序列`, `72. 编辑距离` 一起考虑，因为状态转移方程的思想都很类似。

定义：f(i, j) 为 `s1 in [0...i], s2 in [0..j]` 能够使 s1 和 s2 完全相同所需要操作的最小步数。

直接上代码来体现转移方程：
```c++
int m = s1.size(), n = s2.size();
int f[m + 1][n + 1];
memset(f, 0x3f, sizeof f);
for (int i = 0; i <= m; i ++ )  f[i][0] = i;
for (int j = 0; j <= n; j ++ )  f[0][j] = j;
for (int i = 1; i <= m; i ++ )
    for (int j = 1; j <= n; j ++ )
        f[i][j] = s1[i - 1] == s2[j - 1] ? f[i - 1][j - 1] : min(f[i - 1][j], f[i][j - 1]) + 1; 
return f[m][n];
```

## 634. 寻找数组的错位排列
题意：对一个有序(从小到大排好序)的序列，转化为其中`所有`元素的位置都与原 `有序` `序列` 所处的位置不同，共有多少种排法？即 1 不可处于位置 1，2 不可处于位置 2，3 不可处于位置 3 ：[1, 2, 3] 的错位排列有：[2, 3, 1] 和 [3, 1, 2]

这里直接上递推式代码来理解，就像`343. 整数拆分`一样，要上代码理解：

定义：f(i) 为 `序列元素数目为 i` 的所有错位排列方案数目。

例：[1, 2, 3, 4, 5, 6, 7, ... , j, ... , i] 我们考虑最后一个数字 i ：则划分为两种方案：

1 > 将 i 与 j `交换位置`，且则剩下余下 `i - 2 个` 数 `都处于原位`，所以在`这 i - 2 个数字`中的错位`方案数`为 `f(i - 2)` 所以此时的方案数为 f(i - 2), 而 j in[1 ... i - 1]共 i - 1 个可选择的 j ，通过`数学上的化简`，我们不用遍历 j，直接写为： `(i - 1) * f(i - 2)`

2 > 将 j 放在 i 这个位置上，此时我们可以假设认为：`正确排列为 i 放在 j 这个位置上，则错位排列为：不将 i 放在 j 原来的位置上`，则相当于对前面这 i - 1 个数进行错位排列(只不过此时默认正确顺序是将 i 放在 j 原来的位置上)，所以而 j in [1 ... i - 1]所以方案数目为 `(i - 1) * f(i - 1)`
```c++
int findDerangement(int n) {
    if(n==1) return 0;
    vector<long long > f(n+1, 0);
    f[0] = 1;
    f[1] = 0;
    for(int i = 2; i <= n; ++i)
        f[i] = ((i - 1) * (f[i - 1] + f[i - 2])) % 1000000007;
    return f[n];
}
```
## 1690. 石子游戏 VII
同 `486. 预测赢家` 我们直接利用简化的状态转移表达式：f(i, j)，这个状态的集合表示与预测赢家的状态集合的解释一模一样，只是其转移时候的`分数解释`变化了而已，用一个前缀和来操作`每次分情况讨论`选取的分数就行了。

状态集合表示：

定义`f(i, j)`表示当数组剩下的部分为下标 i 到下标 j 时，即在下标范围`[i, j]`中，`当前玩家与另一个玩家的分数之差的最大值`，注意当前玩家不一定是先手。而是正在取数的当前回合玩家。

由于前面已经分析过了这种状态表示时的转移策略，所以直接上代码：

公式推导：
1. f(l, r) = first - second = nums[l...r - 1] + first - nums[l...r - 1] - second = nums[l...r - 1] - (second - first')。`其中 nums[l...r - 1] 是删除 r 后的得分`
2. 上式中 first' 是原 first 删除了 nums[l...r - 1] 后的 first'，即 `first' = first - nums[l...r - 1]`
3. second - first' = f(l, r - 1)
4. f(l, r) = nums[l...r - 1] - f(l, r - 1)
5. => `f(l, r) = max(nums[l...r - 1] - f(l, r - 1), nums[l + 1...r] - f(l + 1, r))`

```c++
int f[1010], s[1010], n = nums.size();
memset(f, 0, sizeof f); memset(s, 0, sizeof s);
for (int i = 1; i <= n; i ++ )
    s[i] = s[i - 1] + nums[i - 1];
    // 为什么这里不需要这个初始化为 nums[l] 的代码？原因就在前面的分析中，他的分数解释变化了而已，他的分数解释为剩余的元素之和，而当前玩家拿取剩下一个元素的数组后，其数组变为了空数组，所以得分必定为 0，所以初始化为 0 即可。即不需要进行操作。
    // for (int l = 1; l <= n; l ++ ) f[l] = nums[l - 1];       
for (int l = n - 1; l >= 1; l -- )
    for (int r = l + 1; r <= n; r ++ )
        f[r] = max(s[r] - s[l] - f[r], s[r - 1] - s[l - 1] - f[r - 1]);
return f[n];
```

## LCP 30. 魔塔游戏 ： 模拟 + 贪心 + 优先队列
方法：`反悔型模拟` + 贪心


数据结构： 小根堆优先队列

算法：
1. 首先计算是否 `sum(nums)` 是否 < 0 ，如果 < 0 则直接返回 -1
2. 遍历 nums 数组中的元素 e：
    2.1 > hp += e   此时可以保证，如果 e 使得 hp < 1 那么 hp 绝对小于 0， 所以这可以保证下面的` e < 0 这个条件`成立，也就保证了`如过当前 hp < 1 时`，堆里面的元素必不空
    2.2 > if e < 0 then 将 e 加入小根堆队列。
    2.2 > if hp < 1 then 将堆顶元素 t 弹出，并让 hp -= t 相当于把遍历过程中最小的那个值放在最后面反悔。

```c++
typedef long long LL;
int magicTower(vector<int>& nums) {
    // 判断是否能到达目的地
    LL sum = 0;
    for (auto e : nums) sum += e;
    if (sum < 0) return -1;

    // 进行模拟通关，通过小根堆来贪心
    LL hp = 1, n = nums.size(), ans = 0;
    priority_queue<int, vector<int>, greater<int> > q;
    for (int i = 0; i < n; i ++ ) {
        hp += nums[i];
        if (nums[i] < 0) q.push(nums[i]);
        if (hp <= 0) {
            int t = q.top(); q.pop();
            hp += -t;
            ans ++ ;
        }
    }
    return ans;
}
```

## 1696. 跳跃游戏 VI

令 `f(i)` 为当跳到 i 时能获得的最大值，那么动态规划转移方程为：

f(i) = max{f(j)} + nums[i]  // 其中 j 为 i 前面的下标，范围为 k。

朴素写法：直接用朴素代码实现上面的`数学转移方程`，可以用当前的更新后面的，也可以用之前的更新当前的两种思路，以下代码为前者的思路，时间复杂度为 `O(k * n)`
```c++
const int INF = 0x3f3f3f3f;
int maxResult(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> f(n, -INF); f[0] = nums[0];
    for (int i = 0; i < n; i ++ ) 
        for (int j = 1; j <= k && i + j < n; j ++ ) 
            f[i + j] = max(f[i + j], f[i] + nums[i + j]);
    return f[n - 1];
}
```

单调队列优化写法：也可以使用单调队列的代码思想来优化`数学转移方程`的实现，时间复杂度为`O(n)`：

具体来说，利用单调队列来维护 `f[]` 数组(切记不是`nums[]数组`)中的大小为 k 的单调窗口，那么对于枚举区间右端点 i 时，我们可以`用O(1)的时间复杂度`得到该区间的最大值，从而可以得到 `f[i + 1]` 的值：

`f[i + 1] = f[q.front()] + nums[i + 1];` 其中 `f[q.front()]` 为当前大小为 k 的窗口的最大值

代码：
```c++
n = nums.size();
int f[n];
f[0] = nums[0];
deque<int> q;
for (int i = 0; i < n - 1; i ++ ) {
    while (q.size() && q.front() < i - k + 1) q.pop_front();
    while (q.size() && f[q.back()] <= f[i]) q.pop_back();

    q.push_back(i);
    f[i + 1] = f[q.front()] + nums[i + 1];
}
return f[n - 1];
```

## 292. Nim 游戏
可以发现，Nim 游戏使用的是归纳法，也就是和动态规划的本质类似，因为动态规划的本质也是数学归纳法，找通项公式。

1.如果落到先手的局面为「石子数量为1-3」的话，那么先手必胜;
2.如果落到先手的局面为「石子数量为4」的话，那么先手决策完（无论何种决策)，交到后手的局面为「石子数量为1-3」，即此时后手必胜，对应先手必败(到这里我们有一个推论:如果交给先手的局面为4的话，那么先手必败);
3.如果落到先手的局面为「石子数量为5-7」的话，那么先手可以通过控制选择石子的数量，来使得后手处于「石子数量为4」的局面(此时后手必败)，因此先手必胜;
4.如果落到先手的局面为「石子数量为8」的话，由于每次只能选1-3个石子，因此交由后手的局面为5- 7，根据流程3我们知道此时先手必败;

那么一直递推，我们发现，如果 n 可以被 4 整除就必败，如果 n 不能被 4 整除，则必胜。代码如下：
```c++
bool canIwin(int n) {
    return n % 4 != 0;
}
```


## 638. 大礼包 -> 预处理 + 记忆化搜索
这种是类似于小猫爬山型 DFS，不是八皇后型 DFS。

数据结构：略

算法步骤：

1 > 预处理大礼包，我们让包里有产品，且`确实有优惠(即礼包价格比单买价格有优惠)`的礼包保留下来，其他礼包丢弃不用

2 > 进行 DFS 记忆化搜索，让`当前所需要的产品及其数量作为状态集合` 用一个 vector<int> 数组来记录这样一个状态。状态转移方程为：
`memo[cur] = min{不使用大礼包, 购买第 i 个大礼包 + DFS(nxtNeeds) // nxtNeeds 为购买完第 i 个礼包后所得到的状态集合}`


通过代码实现上述`预处理和数学转移方程`

```c++
const int INF = 0x3f3f3f3f;
struct hf{
    size_t operator () (const vector<int> &v) const {
        size_t res = 0;
        for (auto &e : v) 
            res ^= hash<int>()(e) + 0x9e3779b9 + (res << 6) + (res >> 2);
        return res;
    }
};
class Solution {
public:
    unordered_map<vector<int>, int, hf> memo;
    int n;
    vector<vector<int> > a;
    vector<int> price, needs;
    int DFS(vector<int> cur) {
        if (memo.count(cur)) return memo[cur];
        auto &ans = memo[cur]; ans = 0;
        for (int i = 0; i < n; i ++ ) ans += cur[i] * price[i]; // 这里用不购买任何大礼包来初始化 memo[cur]

        // 此 for 循环代码来实现该数学转移方程：`memo[cur] = min{不使用大礼包, 购买第 i 个大礼包 + DFS(nxtNeeds) // nxtNeeds 为购买完第 i 个礼包后所得到的状态集合}`
        for (auto &e : a) {
            vector<int> nxtNeeds;
            for (int i = 0; i < n; i ++ ) {
                if (e[i] > cur[i]) break;
                nxtNeeds.push_back(cur[i] - e[i]);
            }
            if (nxtNeeds.size() == n) ans = min(ans, DFS(nxtNeeds) + e[n]);
        }
        return ans;
    }
    int shoppingOffers(vector<int>& _price, vector<vector<int>>& special, vector<int>& _needs) {
        price = _price; needs = _needs; n = price.size();
        // 这里实现预处理
        for (auto &e : special) {
            int totalCnt = 0, totalPrice = 0;
            for (int i = 0 ; i < n; i ++ ) {
                totalCnt += e[i];
                totalPrice += e[i] * price[i];
            }
            // totalCnt > 0 即礼包里有产品，e[n] < totalPrice 即此礼包比单价购买有优惠
            if (totalCnt > 0 && e[n] < totalPrice) a.push_back(e);  
        }
        return DFS(needs);
    }
};
```



## 2641. 二叉树的堂兄弟节点 II

利用分层式 BFS 遍历模板，由于需要进行两次遍历，所以要`用 vector 代替 queue`，这样能使层次遍历的代码更容易理解，`代码更具有可读性`

数学分析：`x 的孩子节点的所有堂兄弟节点值的和` = `下一层的每个子节点和` - `x 节点的孩子节点的和`

代码实现：利用 vector 式 BFS 模板 + 两次遍历来实现上面这个数学表达式。

我们再此总结一下 queue 模板不好的地方：queue 模板不能让我们看到 队列内部的情况，我们无法利用 for (auto e : q) 这种方式去查看 q 内的元素，所以我们如果使用 vector<TreeNode> 来代替 queue，我们就可以在`不必 pop() 的情况下``访问`队列内部`所有的元素`。这是非常好的东西。

我们如果使用 vector 式 BFS 模板，我们就可以`很方便地`观察到每一层的队列内部情况。而这也是一层一层地打印出 层序遍历的最好的方式。
```c++
TreeNode* replaceValueInTree(TreeNode* root) {
    vector<TreeNode*> q;
    q.push_back(root);
    root->val = 0;
    while (q.size()) {
        vector<TreeNode*> nxt;
        int sum = 0;
        for (int i = 0; i < q.size(); i ++ ) {
            if (q[i]->left) sum += q[i]->left->val;
            if (q[i]->right) sum += q[i]->right->val;
        }
        for (int i = 0; i < q.size(); i ++ ) {
            int childSum = 0;                           // 计算自己的孩子的和
            if (q[i]->left) childSum += q[i]->left->val;
            if (q[i]->right) childSum += q[i]->right->val;
            if (q[i]->left) q[i]->left->val = sum - childSum, nxt.push_back(q[i]->left);
            if (q[i]->right) q[i]->right->val = sum - childSum, nxt.push_back(q[i]->right);
        }
        q = move(nxt);
    }
    return root;
}
```

## 993. 二叉树的堂兄弟节点
如果二叉树中， x 与 y 互为堂兄弟，我们有这个数学意义上的结论：x 与 y 处于同一层，且双亲一定不同。

我们再此总结一下 queue 模板不好的地方：queue 模板不能让我们看到 队列内部的情况，我们无法利用 for (auto e : q) 这种方式去查看 q 内的元素，所以我们如果使用 vector<TreeNode> 来代替 queue，我们就可以在`不必 pop() 的情况下``访问`队列内部`所有的元素`。这是非常好的东西。

代码如下：
```c++
bool isCousins(TreeNode* root, int x, int y) {
    vector<TreeNode*> q;
    q.push_back(root);
    while (q.size()) {
        vector<TreeNode*> nxt;
        TreeNode *px, *py;
        int flagX = 0, flagY = 0;
        for (int i = 0; i < q.size(); i ++ ) {
            if (q[i]->left && q[i]->left->val == x) {px = q[i], flagX = 1;}
            if (q[i]->right && q[i]->right->val == x) {px = q[i], flagX = 1;}
            if (q[i]->left && q[i]->left->val == y) py = q[i], flagY = 1;
            if (q[i]->right && q[i]->right->val == y) py = q[i], flagY = 1;
            if (q[i]->left) nxt.push_back(q[i]->left);
            if (q[i]->right) nxt.push_back(q[i]->right);
        }
        if (flagY == 1) cout << flagX << endl;
        if (flagX == 1 && flagY == 1) {
            if (px != py) return 1;
            else return 0;
        }
        if (flagX == 1 || flagY == 1) return 0;
        q = nxt;
    }
    return 0;
}
```

## 646. 最长数对链
可以利用`排序 + 动态规划`或者利用`排序 + 贪心`来做。这里介绍`排序 + 贪心`的做法，与安排会议问题完全一样
```c++
bool cmp(const vector<int> &a, const vector<int> &b){
    return a[1] < b[1];
}
int findLongestChain(vector<vector<int>>& pairs) {
    sort(pairs.begin(), pairs.end(), cmp);
    int curR = pairs[0][1], n = pairs.size(), ans = 1;
    for (int i = 1; i < n; i ++ ) 
        if (curR < pairs[i][0]) { ans ++ ; curR = pairs[i][1]; }
        else continue;
    return ans;
}
```

## 647. 回文子串
推荐一个利用中心扩散法的题目，和一个判断回文串的方式：
```c++
bool check(string &s, int l, int r) {
    while (l < r) {
        if (s[l] != s[r]) return 0;
        l ++ , r -- ;
    }
    
}


int countSubstrings(string s) {
    int n = s.size(), ans = 0;
    for (int i = 0; i < n; i ++ ) {
        int l = i, r = i;
        while (l >= 0 && r < n) {   // 中心扩散法：奇长度中心
            if (s[l] == s[r]) ans ++ , l -- , r ++ ;
            else break;
        }

        if (i == n - 1) break;      // 最后一个节点不能算入偶节点中心。

        l = i, r = i + 1;       // 偶长度中心
        while (l >= 0 && r < n) {
            if (s[l] == s[r]) ans ++ , l -- , r ++ ;
            else break;
        }
    }
    return ans;
}
```

## 650. 两个键的键盘  数学 + 动态规划
这里的`数学思想`还是挺重要的，利用了`质因数`的思想结合动态规划，挺好的一种`数学`思想。

我们先定义：f(i)为打印出 i 个 A 的最少操作次数。那么首先，我们必须要知道，不是任何 f(j) 都能转移到 f(i) 的。如果上一次屏幕上的 A 的数量为 j，且粘贴板内的 A 的数目有 k，那么 `j + n * k 必须等于 i`：我们来分析这个 j 与 k：由于每个数目 i 只能是最后一次 copy ALL 操作和 n次(若干次) paste 操作得到。我们 `令这个 j 是最后一次 copy ALL 操作时的屏幕上的 A 的数目` 那么此时 k 一定等于 j。由这个道理，我们得知这个 j 一定是 i 的因数，即 `j + n * j = (n + 1) * j 必须等于 i` 。由于刚开始屏幕上已经有 j 个 A 了，所以`粘贴的次数`就是 `n = (i / j) - 1`, 那么就可以得到 i 个 A 了。而加上 copy ALL 一次的次数总共 `n + 1 = i / j` 次数。那么状态转移方程很容易就可以得到：

`f(i) = min{f(j) + i / j} 其中 j 为 i 因数。`

代码实现：如果朴素实现上述数学思想的话，那么要 `O(n^2)` 的时间复杂度，但是按照我们在算法课上的`试除法`来说，我们能够得到 O(n*√n) 的代码复杂度，即对于每个因数 j <= √i 都有一个对应的 i / j >= √i 因数与其对应。但我觉得在考场上可能还是老老实实的用 O(n^2) 的方法来实现。

```c++

int minSteps(int n) {
    int f[n + 1]; memset(f, 0x3f, sizeof f); f[1] = 0;
    for (int i = 1; i <= n; i ++ )       // 这里第二层循环就是复杂度减少到了 √n
        for (int j = 1; j <= i / j; j ++ )
            if (i % j == 0)     // 我们只能从 i 的因子这个位置转移过来
                f[i] = min({f[i], i / j + f[j], f[i / j] + j});     // 其中 i 一共有两个因数：分别为： j 和 i / j
    return f[n];
}
```

### 我们再次利用数学优化，将上述思想其优化到 √n:
这个数学优化实在是有点难想，考试绝对想不出，但是平时需要去积累。

由于每个 f[i] 都是从 f[j] + j 转移过来的，其中 j 可以整除 i -> 那么 j 同样是从 f[k] 转移过来的，其中 k 可以整除 j 。那么依次分解 i 的因数，我们可以发现一个规律：f[i] = x1 + x2 + x3 + ... 也就是说，一定是全部拆成质因数才能让结果更好，具体的证明可以看`宫水三叶`结合`官方题解`来理解。

代码：
```c++
int minSteps(int n) {
    int ans = 0; 
    for (int i = 2; i <= n / i; i ++ )
        while (n % i == 0) n /= i, ans += i;
    if (n > 1) ans += n;
    return ans;
}
```

## 651. 四个键的键盘
首先，我们来分析此题与题 `650. 两个键的键盘` 的区别。这个题就是比较常规的动态规划题了，比 `650. 两个键的键盘` 要简单很多，就是最常规的动态规划，根本没有 `650. 两个键的键盘` 的数学层面上的限制，它可以从`更多`的之前的`状态集合`进行转移过来。

令 f(i) 为第 i 次按下键时，最屏幕上最多可以显示 `f(i)` 个 A。那么它就完全没有 `650. 两个键的键盘` 这个题目数学上的种种限制，直接可以用很朴素的两层循环来处理。这个题目的状态转移的核心模拟只能为两种情况：

1. 直接按一个 A 键，让一个字母 A 打印上屏幕的操作。
2. 如同题 `650. 两个键的键盘` 的模拟行为，其最后一次操作一定为 ctrA + ctrC + n * ctrV -> 一次全选 + 一次复制 + 若干次(n次)paste

那么我们就直接能够通过上面的分类讨论的数学思想来写出代码：
```c++
int maxA(int n) {
    int f[n + 1]; memset(f, 0, sizeof f);
    f[0] = 0;
    for (int i = 1; i <= n; i ++) {
        // 按 A 键
        f[i] = f[i - 1] + 1;
        for (int j = 1; j <= i - 2; j ++) { // i - 2, i - 1, i => ctrlA, ctrlC, ctrlV
            // 设从 j 开始进行  `一次全选 + 一次复制 + 若干次(n次)paste` 的操作，则
            // paste 的次数 n = i - j + 1 - 2 = i - j - 1；那么屏幕上的 A 个数就是 i - j - 1 + 1 = i - j 倍的 f[j - 1] 个 A
            f[i] = max(f[i], f[j - 1] * (i - j));
        }
    }
    // n 次按键之后最多有几个 A？
    return f[n];
}


```

## 236. 二叉树的最近公共祖先
注意，这里的思路与倍增算法不同，倍增算法处理的场景是`同一颗` + `多叉树` + `大量查询`，所以它预处理用时 O(n), 查询一次用时 O(logn), 查询 n 次用时 O(n * logn)；而此题的场景是`一棵树` + `仅一次查询`，所以这里直接利用 DFS 来查询一次，用时 O(n) 即可。

我们利用更加有利于写出代码的数学定义形式来定义一个二叉树的公共祖先：

有利于代码实现的最近公共祖先的数学`定义`:设节点 root 为`节点 p, q 的某公共祖先`，满足它的深度尽可能大(即高度尽可能的低)。


根据上述数学定义，我们很容易写出算法:
我们递归遍历二叉树:
1. 如果当前节点为 `空` 或者等于 p 或者 q，则返回当前节点;   // 即此时的情况为：当前节点为空 或者 LCA 正好是 q 或者 p
2. 否则，我们递归遍历左右子树，将返回的结果分别记为 left 和 right。如果 left 和 right 都不为空，则`说明 p 和 q 分别在左右子树中`，所以当前节点即为最近公共祖先; 如果 left 和 right 中`只有一个不为空`，则说明这两个节点都在`同一颗子树`中，那么我们返回在这颗子树下 `最深` 的公共祖先，而`不是`返回这颗`子树的根节点`，这个非常关键。

```c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == NULL || root == p || root == q) return root;
    // 这里使用`递归调用` LCA() 函数体现了去寻找 `高度最低，深度最大` 的公共祖先
    // 先去左子树寻找最近公共祖先，再去右子树中寻找最近公共祖先，也就是实现了去找最深的 LCA
    TreeNode *left = lowestCommonAncestor(root->left, p, q);        // 去左子树寻找 `最深的` 公共祖先
    TreeNode *right = lowestCommonAncestor(root->right, p, q);      // 去右子树寻找 `最深的` 公共祖先
    if (left && right) return root; // 如果 left 和 right 都不为空，则说明 p 和 q 分别在左右子树中，因此当前节点即为最近公共祖先
    return left ? left : right;     // 如果只有一个子树能找到，则说明 这两个节点都在同一颗 子树下，那么我们返回在 那颗子树下能找到的最深的根节点，而不是返回这个子树的根节点。
}
```


## 673. 最长递增子序列的个数
1. 动态规划写法：

对于每个下标 i，记录下每个以 i 元素为结尾，最长的递增子序列的长度 f(i)，同时记录下以 i 元素为结尾的最长递增子序列的数量 g(i)。这样就很容易得到状态转移方程了：具体请见代码

易错点：
1. 我们在找到一条更长的路径时，必须是：g[i] = g[j], 而不是 g[i] = 1。这是最关键的。
```c++
int findNumberOfLIS(vector<int>& nums) {
    int n = nums.size();
    int f[n], g[n];
    for (int i = 0 ; i < n; i ++ ) f[i] = g[i] = 1;
    for (int i = 1; i < n; i ++ )
        for (int j = 0; j < i; j ++ ) 
            if (nums[j] < nums[i]) {
                if (f[i] < f[j] + 1) f[i] = f[j] + 1, g[i] = g[j];  // 如果能得到更长的递增子列则重新赋值 g[i] = g[j]
                else if (f[i] == f[j] + 1) g[i] += g[j];            // 如果在 j 处能得到长度相同的递增子列则 g[i] += g[j]
            }
    int ML = 0, ans = 0;
    for (int i = 0; i < n; i ++ )           
        ML = max(ML, f[i]);
    for (int i = 0; i < n; i ++ )
        if (ML == f[i]) ans += g[i];
    return ans;
}
```
## 678. 有效的括号字符串
栈：这个思路需要背诵下来，想是想不出来的

要创建两个栈：一个`栈 skt1` 存放 '(' ，一个`栈 stk2` 存放 '*'。然后直接模拟什么时候操作合法：
1. 如果为 '(' 则直接将其下标 id1 入 stk1 
2. 如果为 '*' 则直接将其下标 id2 入 stk2
3. 如果为 ')' 则先从 stk1 内用 '(' 去匹配，如果 stk1 为空，才从 stk2 内用 '*' 去匹配。如果两个栈中都为空了则返回 false。

按上面的那个方法，如果合法的话那么最后`右括号一定会被匹配完毕`，而 stk1 和 stk2 可能非空。那么此时就要用 * 号来去匹配 '('，直到 stk1 内的 '(' 变空。且根据实际情况可知，越靠近栈顶的位置，其 id 的值越大。遍历 stk1 的栈顶：'('
1. 如果 stk2 的栈顶 '*' 的下标在 stk1 '(' 的右边则弹出两者的栈顶
2. 否则匹配不了，返回 false 。

如果最后 stk1 为空，则返回 ture，而 stk2 内的 * 此时可以全部视为空
```c++
bool checkValidString(string s) {
    stack<int> stk1, stk2;
    int n = s.size();
    for (int i = 0; i < n; i ++ ) {
        if (s[i] == '(') stk1.push(i);
        else if (s[i] == '*') stk2.push(i);
        else {
            if (stk1.size()) stk1.pop();
            else if (stk2.size()) stk2.pop();
            else return 0;
        }
    }
    while (stk1.size() && stk2.size()) {
        if (stk1.top() < stk2.top()) stk1.pop(), stk2.pop();
        else return 0;
    }
    return stk1.empty();
}
```

## 688. 骑士在棋盘上的概率
其状态的定义与 `576. 出界的路径数` 类似，直接上代码即可：

```c++
int dirs[8][2] = { {1, 2}, {1, -2}, {2, 1}, {2, -1}, {-1, 2}, {-1, -2}, {-2, 1}, {-2, -1} };
double memo[27][27][110];
int n;
double DFS(int x, int y, int k) {
    if (x >= n || x < 0 || y >= n || y < 0) return 0;
    if (k == 0) { return 1;}
    if (memo[x][y][k] != -1) return memo[x][y][k];

    double &ans = memo[x][y][k]; ans = 0;
    for (int i = 0; i < 8; i ++ ) {
        int nx = x + dirs[i][0], ny = y + dirs[i][1];
        ans += DFS(nx, ny, k - 1) / 8;
    }
    return ans;
}
double knightProbability(int _n, int k, int row, int column) {
    n = _n; 
    for (int i = 0; i < 27; i ++ )
        for (int j = 0; j < 27; j ++ )
            for (int k = 0; k < 110; k ++ )
                memo[i][j][k] = -1;
    return DFS(row, column, k);
}
```

## 698. 划分为k个相等的子集
直接利用状态压缩与记忆化搜索。DFS(step, cur, state) 代表现在已经完美的填满了 step 个子集，当子集的元素之和为 cur，已经使用的元素记录在 state 中。

在代码实现时，可以将 state 放在全局变量上，而不是作为参数传递来简化代码。
```c++
int memo[1 << 18];
vector<int> nums;
int k, tar, n, state;
bool DFS(int step, int cur) {
    if (step == k) return 1;
    if (memo[state] != -1) return memo[state];

    auto &ans = memo[state]; ans = 0;
    for (int i = 0; i < n; i ++ ) {
        if (((state >> i) & 1) == 1 || cur + nums[i] > tar) continue;
        if (cur + nums[i] == tar) {
            state |= 1 << i;
            ans = ans || DFS(step + 1, 0);
            state &= ~(1 << i);
        }
        else if (cur + nums[i] < tar) {
            state |= 1 << i;
            ans = ans || DFS(step, cur + nums[i]);
            state &= ~(1 << i);
        }
        if (ans) return ans;
    }
    return ans = 0;
}
bool canPartitionKSubsets(vector<int>& _nums, int _k) {
    nums = _nums, k = _k, n = nums.size(), state = 0;
    memset(memo, -1, sizeof memo);
    int sum = 0;
    for (auto e : nums) sum += e;
    if (sum % k) return 0;
    tar = sum / k;
    return DFS(0, 0);
}  
```

## 712. 两个字符串的最小ASCII删除和
这题最重要的是初始化工作，因为这个初始状态如果不初始化那么开始的状态一定会导致错误的转移。

这是因为在状态转移的循环中，要用到的初始边界条件是必须经过初始化的，而不是 0 。

```c++
int minimumDeleteSum(string s1, string s2) {
    int m = s1.size(), n = s2.size();
    int f[m + 1][n + 1];
    memset(f, 0, sizeof f);
    for (int i = 1; i <= m; i ++ ) f[i][0] = f[i - 1][0] + s1[i - 1];
    for (int j = 1; j <= n; j ++ ) f[0][j] = f[0][j - 1] + s2[j - 1];

    // 可以看到：用到的初始边界条件是必须经过初始化的，而不是 0 。
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (s1[i - 1] == s2[j - 1]) f[i][j] = f[i - 1][j - 1];
            else f[i][j] = min(f[i - 1][j] + s1[i - 1], f[i][j - 1] + s2[j - 1]);
    return f[m][n];
}
```

## 714. 买卖股票的最佳时机含手续费
直接上代码，经典的买卖股票：
```c++
int maxProfit(vector<int>& prices, int fee) {
    const int INF = 0x3f3f3f3f;
    int n = prices.size();
    int f[n + 1][2];
    f[0][0] = 0;
    f[0][1] = -INF;
    for (int i = 1; i <= n; i ++ ) {
        f[i][0] = max(f[i - 1][0], f[i - 1][1] + prices[i - 1]);
        f[i][1] = max(f[i - 1][1], f[i - 1][0] - fee - prices[i - 1]);
    }
    return f[n][0];
}
```

## 718. 最长重复子数组
这个和 `712. 两个字符串的最小ASCII删除和` 的显著区别就是这个不需要进行特别的初始化，因为 0 正好是边界条件的初始化。

所以此题要比 `712. 两个字符串的最小ASCII删除和` 要简单。两者的思路差不多，代码如下：

```c++
int findLength(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size();
    int f[m + 1][n + 1]; memset(f, 0, sizeof f);
    int ans = 0;
    for (int i = 1; i <= m; i ++ ) {           
        for (int j = 1; j <= n; j ++ ) {
            if (nums1[i - 1] == nums2[j - 1]) f[i][j] = f[i - 1][j - 1] + 1;
            else f[i][j] = 0;
            ans = max(f[i][j], ans);
        }
    }
    return ans;
}
```

## 740. 删除并获得点数
方法一：统计数量 + 打家劫舍

所以打家劫舍的精华所在就是揭示了相邻元素和相隔一个元素之间的那种选取关系，如果问题能抽象出隔元素取数的模型则可以利用打家劫舍的思想。

算法：收集 `nums[]` 内的所有元素并记录下其数量，并将其按照打家劫舍的思想来处理。
```c++
const int N = 1e4 + 10;
int deleteAndEarn(vector<int>& nums) {
    int a[N]; memset(a, 0, sizeof a);
    int n = 0;
    for (auto e : nums) a[e] ++ , n = max(n, e);
    int f[N]; memset(f, 0, sizeof f); f[1] = a[1];
    for (int i = 2; i <= n; i ++ )
        f[i] = max(f[i - 1], f[i - 2] + a[i] * i);
    return f[n];
}

```
## 764. 最大加号标志
此题与 `221. 最大正方形` 是一样注重模拟的题目，所以我们需要积累这种模拟的经验：

该题需要分别处理出 矩阵中 每个点分别在 `上下左右` 拥有的`最大连续 1 延申距离` 我称之为 `某一点的臂展(其长度包括该店)`

然后答案就是这个点`分别沿四个方向上臂展的最小值`。遍历出每个点然后求出最大值。

```c++
int orderOfLargestPlusSign(int n, vector<vector<int>>& mines) {
    int maze[n + 2][n + 2]; memset(maze, 0, sizeof maze);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            maze[i][j] = 1;
    for (auto e : mines)
        maze[e[0] + 1][e[1] + 1] = 0;
    int f[4][n + 2][n + 2]; memset(f, 0, sizeof f);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ ) {
            f[0][i][j] = maze[i][j] ? f[0][i][j - 1] + 1 : 0;
            f[1][i][j] = maze[i][j] ? f[1][i - 1][j] + 1 : 0;
        }
    
    for (int i = n; i >= 1; i -- )
        for (int j = n; j >= 1; j -- ) {
            f[2][i][j] = maze[i][j] ? f[2][i][j + 1] + 1 : 0;
            f[3][i][j] = maze[i][j] ? f[3][i + 1][j] + 1 : 0;
        }
    int ans = 0;
    for (int i = 1; i <= n; i ++ ) 
        for (int j = 1; j <= n; j ++ )
            if (maze[i][j])
                ans = max(ans, min({f[0][i][j], f[1][i][j], f[2][i][j], f[3][i][j]}));
    return ans;
}
```

## 746. 使用最小花费爬楼梯
这个题目可以很好地来练习从当前的状态去更新若干下一次的状态，也就是和之前的状态转移方式略有不同，但是数学本质一样的，只不过代码实现方式有区别而已，哪样简便用哪个。
```c++
int minCostClimbingStairs(vector<int>& cost) {
    int n = cost.size();
    int f[n + 2]; memset(f, 0x3f, sizeof f); 
    f[0] = f[1] = 0;
    for (int i = 0; i < n; i ++ )
        f[i + 1] = min(f[i + 1], f[i] + cost[i]),
        f[i + 2] = min(f[i + 2], f[i] + cost[i]);
    return f[n];
}
```

## 787. K 站中转内最便宜的航班
直接用 Bellman Ford 来解决这个问题，如果深刻地理解了 Bellman Ford  那么这个问题就是很简单了。

之所以归纳到动态规划，实际上本质来说，这个 Bellman Ford 算法本质上算是一个`降维空间复杂度的动态规划`，即将 `f[t][y]: 中转 t 次，目的地为 y 的最小花费` 降维到了：`f[y]`。为了实现这一个降维空间复杂度，将第一维度的 k 降维成 1，所以使用了 `backup[]` 来相当于做为 `f[t - 1][y]` 转移到 `f[t][y]` 的一个转移中介。至于完整的未降维的动态规划做法可参考官方题解，根据刚才的数学分析可知，其实它就是一个弱鸡版的 Bellman Ford。

```c++
int d[110], backup[110];
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    memset(d, 0x3f, sizeof d);
    d[src] = 0;
    for (int i = 0; i <= k; i ++ ) {    // 这里的`题目要求`稍微和 Bellman Ford 算法有些区别，他是 k 个中转站，所以也就是需要迭代 k + 1 条边
        memcpy(backup, d, sizeof d);
        for (int j = 0; j < flights.size(); j ++ ) {
            int x = flights[j][0], y = flights[j][1], z = flights[j][2];
            d[y] = min(d[y], backup[x] + z);
        }
    }
    if (d[dst] == INF) return -1;
    else return d[dst];
}
```

## 790. 多米诺和托米诺平铺
状态机DP：

定义 f(i, j) 为瓷砖铺到第 i 列时，且第 i 列为 j 状态时的所有方案数。其中 j 只有四种状态，故 j in [0, 4)。

这四种状态分别为：
``` 
0   1   2   3
O   X   O   X
O   O   X   X
上图中 O 代表这个位置为空，而 X 代表这个位置已经被填上了瓷砖。
```

那么这三种状态如何转移呢？
0. 第 0 个状态是最容易转移的：`f[i][0] = f[i - 1][3]` 这个非常容易理解，当前列瓷砖全空的状态肯定从上一列全满的状态 3 转移过来的。
1. 仔细观察后发现，`当前列 i` 第 1 种状态可以从 `上一列 i - 1` 的第 2 种状态转移过来也可以从第 0 种状态转移过来。`分别`是0.模拟横着在第一行放一个`多米诺`瓷砖，和2.模拟在上一列用合适的方法放一个`托米诺`瓷砖：`f[i][1] = f[i - 1][0] + f[i - 1][2]`
2. 同 2 中的状态模拟讲解，我们得到这个状态 3 的转移方程：`f[i][2] = f[i - 1][0] + f[i - 1][1]`
3. 最后一个状态可以从第 i - 1 列的四种状态都可以转移过来：分别是：0.模拟放两个横向的多米诺，1.模拟用合适的方法放一个托米诺，2.同 1 ，模拟用合适的方法放一个托米诺，3.模拟竖着放一个多米诺。 那么转移方程为：`f[i][3] = f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3]`。 `注`：这里可能你有疑问：为什么为什么不能从 i - 1 的 0 状态用竖着放两个多米诺来模拟？因为这一种方案和第 3 个状态的转移重复了，第三个状态就是相当于在 `f[i -1][0]` 的基础上先竖着一个多米诺，所以竖着放两个多米诺相当于 `f[i - 1][3]` 这个转移模拟。如果你想用 `f[i - 1][0]` 竖着放两个多米诺这种模拟方法，那么必须去除第 3 中转移状态模拟，加上这种模拟方法也是没有问题的。所以说这个东西必须要记住，不然非常容易搞得头脑发昏。

代码如下：
```c++
int numTilings(int n) {
    LL f[n + 1][4]; memset(f, 0, sizeof f);
    f[1][0] = 1, f[1][3] = 1;
    for (int i = 2; i <= n; i ++ ) {
        f[i][0] = f[i - 1][3];
        f[i][1] = (f[i - 1][0] + f[i - 1][2]) % MOD;
        f[i][2] = (f[i - 1][0] + f[i - 1][1]) % MOD;
        f[i][3] = (f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3]) % MOD;
    }
    return f[n][3];
}
```

## 750. 角矩形的数量      ->      动态规划 + 前缀信息

此题就是 `前缀信息 + 动态规划` 的典型题目：

令 `f[r][c1][c2]` 为`第 r 行以及之前`，其 c1 和 c2 同时为 1 的这样的情况的所有数量。

那么 如果当前行能够使得满足 c1 和 c2 同时都为 1 则有：`ans += (mat[r][c1] && mat[r][c2]) ? f[r - 1][c1][c2] : 0` 和 `f[r][c1][c2] = (mat[r][c1] && mat[r][c2]) ? f[r - 1][c1][c2] + 1 : f[r - 1][c1][c2]`

我们直接利用前缀和的性质省略掉第一维度 r，也就是说在循环时，`物理上`隐藏了 r 这一维度，但是`含义上`当外层循环为 r 时，还是包含着 r 这一维度：

```c++
int countCornerRectangles(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size(), ans = 0;
    int f[m][n]; memset(f, 0, sizeof f);
    for (int r = 0; r < m; r ++ )
        for (int c1 = 0; c1 < n; c1 ++ )
            if (mat[r][c1])
                for (int c2 = c1 + 1; c2 < n; c2 ++ )
                    if (mat[r][c2]) ans += f[c1][c2], f[c1][c2] ++ ;
    return ans;
}
```

## 792. 匹配子序列的单词数

暴力方法：不会暴力还优化个屁：暴力判断每个字符串是否是主串的子序列
```c++
int f[60][50010];
bool check(string &e, string &s) {
    if (e.size() > s.size()) return 0;
    int m = e.size(), n = s.size();
    memset(f, 0, sizeof f);
    for (int i = 1; i <= m; i ++ ) 
        for (int j = 1; j <= n; j ++ )
            if (e[i - 1] == s[j - 1])
                f[i][j] = f[i - 1][j - 1] + 1;
            else 
                f[i][j] = max(f[i][j - 1], f[i - 1][j]);
    return f[m][n] == m ;
}
int numMatchingSubseq(string s, vector<string>& words) {
    int ans = 0;
    for (auto &e : words) {
        if (check(e, s))
            ans ++ ;
    }
    return ans;
}
```

优化暴力中的 check() : 二分查找

方法二：位置数组 + 动态规划

在 `514. 自由之路` 中，我们对`位置数组`进行过详细的分析，即对字符串 s 记录下它的每个字符的位置：`pos[s[i] - 'a'].push_back(i);`

然后进行二分查找来加速寻找。这里有一个查找模板，可以结合注释记一下：

```c++
bool check (string &s) {
    int val = -1;
    for (auto e : s) {
        auto &t = pos[e - 'a'];                 // t 是下标数组，即查找下一个对应字符的下标位置在原 s 的哪里
        int i = upper_bound(t.begin(), t.end(), val) - t.begin(); // 这里直接将查找答案化为 int 型 j，而不是迭代器 it，这样代码更好看且方便
        if (i == t.size()) return 0;            // 这里就可以直接利用 j 和 t.size() 来进行判断是否查找成功了。
        val = t[i];
    }
}

```
完整代码：
```c++
vector<int> pos[26];
bool check (string &s) {
    int val = -1;
    for (auto e : s) {
        auto &t = pos[e - 'a'];                 // t 是下标数组，即查找下一个对应字符的下标位置在原 s 的哪里
        int i = upper_bound(t.begin(), t.end(), val) - t.begin(); // 这里直接将查找答案化为 int 型 j，而不是迭代器 it，这样代码更好看且方便
        if (i == t.size()) return 0;            // 这里就可以直接利用 j 和 t.size() 来进行判断是否查找成功了。
        val = t[i];
    }
    return 1;
}
int numMatchingSubseq(string s, vector<string>& words) {
    for (int i = 0; i < s.size(); ++i) pos[s[i] - 'a'].emplace_back(i);     // 构建 pos 数组，记录每个字符串对应的下标
    int ans = 0;
    for (auto& w : words)       // 
        if (check(w))
            ans ++ ;
    return ans;
}
```



方法二：分桶

比如对于 `words = ["a", "bb", "acd", "ace"]`，我们得到以下的分桶结果：
```
a: ["a", "acd", "ace"]
b: ["bb"]
```
我们用队列 queue 这个数据结构来模拟桶，然后总共有 26 个桶，用数组 `arr[26]` 来表示。

然后遍历字符串 s，针对 s 中的每个字符 s[i] 去找到对应的桶：`arr[s[i] - 'a']`。然后去除该桶内每个`元素(一个字符串)`的第一个字符，如果去除完后对应的字符串大小为 0 则直接 ans ++ 。否则将去除完后对应的字符串`转移到另外的对应桶中`。例，如果上述的 a 被匹配完后，则：ans ++ ，且 acd 和 ace 将被转移到 `c 对应的桶中` 如下图：
```
c: ["cd", "ce"]
b: ["bb"]
```
代码：
```c++
int numMatchingSubseq(string s, vector<string>& words) {
    queue<string> arr[26];
    int ans = 0;
    for (auto e : words) arr[e[0] - 'a'].push(e);
    for (auto &e : s) {
        auto &q = arr[e - 'a'];
        for (int i = q.size(); i; i -- ) {
            auto t = q.front(); q.pop();
            if (t.size() == 1) ans ++ ;
            else arr[t[1] - 'a'].push(t.substr(1));
        }
    }
    return ans;
}
```

在实践中，我们可以只在队列中存储对应 `模式串 word` 遍历到的下标 j 即可。而为了去分辨是哪个 word ，只需要再对其进行编号 `word = words[i]` 即可。
```c++
typedef pair<int, int> PII;
int numMatchingSubseq(string s, vector<string>& words) {
    queue<PII> arr[26];
    for (int i = 0; i < words.size(); i ++ ) 
        arr[words[i][0] - 'a'].push({i, 0});
    int ans = 0;
    for (auto e : s) {
        auto &q = arr[e - 'a'];
        for (int k = q.size(); k; k -- ) {
            auto [i, j] = q.front(); q.pop();
            if (j + 1 == words[i].size()) ans ++ ;
            else arr[words[i][j + 1] - 'a'].push({i, j + 1});
        }
    }
    return ans;
}
```



## 799. 香槟塔
方法：基础`数学物理知识`模拟 + 动态规划

此题是一个典型的用当前状态来更新下一状态的动态规划题目，而不是当前状态由上一状态转移过来的题

1. 先明确一个几何知识：每增加一层，杯子数量就会增加 1 ，且杯子之间相邻。

2. 再明确一个物理知识：当一个杯子满了之后，它流向的左右两边的水是相同。即增量相同。

由 <1> 可知，我们可以用两个 `vector<int> f(row), nxtf(row + 1)` 来`分别`模拟`上层与下层`的杯子，并且由几何相邻得对于当前第 i 列，有： `f(i)` 对应的下面两个杯子为 `nxtf(i) 与 nxt f(i + 1)` 。

由 <2> 可知，转移方程应该这样写：遍历当前行的每一列 i 有：当 `当前行且当前列` 杯子的`水体积 > 1 后`，就会流出到下一行的相邻两个杯子，根据上面分析的两个知识，可写出模拟的代码：`if f(i) > 1 then nxtf(i) += (f(i) - 1) / 2, nxtf(i + 1) += (f(i) - 1) / 2`

而由于杯子所获得的水的体积 = 1 后，即使再流入，它也不会再增加了，所以每个杯子的容量可写为：`min{1, f(i)}`;

根据以上分析，我们来写出代码：

```c++
double champagneTower(int poured, int query_row, int query_glass) {
    vector<double> f(1, poured);
    for (int i = 1; i <= query_row; i ++ ) {
        vector<double> nef(i + 1, 0);
        for (int j = 0; j <= i; j ++ ) {
            if (j < i) nef[j] += max(0.0, (f[j] - 1) / 2);          // j 可以从 j 列处获得上一层的酒水
            if (j > 0) nef[j] += max(0.0, (f[j - 1] - 1) / 2);      // j 还可以从 j - 1 列处获得上一层的酒水
        }
        f = move(nef);
    }
    return f[query_glass] >= 1 ? 1 : f[query_glass];
}
```
## 808. 分汤
此题与 `688. 骑士在棋盘上的概率` 和 `576. 出界的路径数` 的记忆化搜索方法有异曲同工之妙，仔细体会它们之间的联系不难写出代码：

```c++
typedef pair<int, int> PII;
struct hf {
    size_t operator () (const PII &p) const {
        return hash<int>()(p.first) ^ hash<int>()(p.second);
    }
};
const int dirs[4][2] = { {100, 0}, {75, 25}, {50, 50}, {25, 75} };
unordered_map<PII, double, hf> memo;
double DFS(int n1, int n2) {
    if (n1 <= 0 && n2 > 0) return 1;
    else if (n1 <= 0 && n2 <= 0) return 0.5;
    else if (n1 > 0 && n2 <= 0) return 0;
    if (memo.count({n1, n2})) return memo[{n1, n2}];

    auto &ans = memo[{n1, n2}]; ans = 0;
    for (int i = 0; i < 4; i ++ ) {
        int nen1 = n1 - dirs[i][0], nen2 = n2 - dirs[i][1];
        ans += DFS(nen1, nen2) / 4.0;
    }
    return ans;
}
double soupServings(int n) {
    if (n >= 4475) return 1.0;  // 这个地方必须优化，因为当 n >= 4475 时概率就趋近于 1 了，这个想不到也没有办法。
    return DFS(n, n);
}
```

## 801. 使序列递增的最小交换次数
此题 `模拟和数学的性质` 是去定义状态集合的关键。

首先，若要满足这两个数组是合法数组，即无论如何操作都能保证能凑出一种使这两种数组都递增的形式。则要满足以下两个情况中的一种。否则一定不会在测试用例中
1. nums1[i - 1] < nums1[i] && nums2[i - 1] < nums2[i]
2. nums1[i - 1] < nums2[i] && nums2[i - 1] < nums1[i]

如果能理清楚这一点就能够更好理解状态转移的方程了。但是如何去定义状态集合呢？我们利用经验来定义：
设 `f(i, 0)` 为到当前位置 i 为止，使得满足数组 nums1 与 nums2 都严格递增且当前位置 i `不进行`交换操作的最少操作数。
设 `f(i, 1)` 为到当前位置 i 为止，使得满足数组 nums1 与 nums2 都严格递增且当前位置 i `要进行`交换操作的最少操作数。

case 1: 如果满足情况 1 而不满足情况 2 时，举例：
```
4 7
9 13    // 具体不满足的形式为：`nums2[i - 1] < nums1[i]`
```
则状态转移方程为： `f(i, 0) = f(i - 1, 0); f(i, 1) = f(i - 1, 1) + 1;` 即当前 i 位置不交换则从 i - 1 位置的`不交换情况 0 状态`转移过来，且不需要操作；当前 i 位置若要交换，则前一个位置必须`也必须要交换(否则将使结果非法)`，则从 i - 1 位置的`交换的情况 1 状态`转移过来，且操作数要 + 1.

case 2: 如果只满足情况 2 而不满足情况 1 时，举例：
```
7 5     // 具体不满足的形式为： `nums1[i - 1] < nums1[i]`
4 9
```
则状态转移方程为：`f(i, 0) = f(i - 1, 1); f(i, 1) = f(i - 1, 0) + 1` 分析同 case 1。

case 3: 同时满足情况 1 与 情况 2：
```
4 8
3 9
```
则状态转移方程为：`f(i, 0) = min{f(i - 1, 0), f(i - 1, 1)}; f(i, 1) = min{f(i - 1, 0), f(i - 1, 1)} + 1` 这个就很好理解了：交换不交换都是可以的，`当前位置 i 的两种状态`分别从`上一个位置 i - 1 处的较小的两者的那个状态`转移过来即可。

在代码实现时，分情况讨论的 case 的顺序要改一下，先讨论最后一种 case3 再讨论 case1 和 case2 ：
```c++
int minSwap(vector<int>& nums1, vector<int>& nums2) {
    int n = nums1.size();
    int f[n][2]; f[0][0] = 0, f[0][1] = 1;
    for (int i = 1; i < n; i ++ )
        if ((nums1[i - 1] < nums1[i] && nums2[i - 1] < nums2[i]) && (nums1[i - 1] < nums2[i] && nums2[i - 1] < nums1[i]))
            f[i][0] = min(f[i - 1][0], f[i - 1][1]), f[i][1] = min(f[i - 1][0], f[i - 1][1]) + 1;
        else if (nums1[i - 1] < nums1[i] && nums2[i - 1] < nums2[i])
            f[i][0] = f[i - 1][0], f[i][1] = f[i - 1][1] + 1;
        else 
            f[i][0] = f[i - 1][1], f[i][1] = f[i - 1][0] + 1;
    
    return min(f[n - 1][0], f[n - 1][1]);
}
```

## 813. 最大平均值和的分组
这个题目非常好，和 `887. 鸡蛋掉落` 的`状态定义和状态集合转移方式` 给了我们全新的思考模式，以前从未见过：

直接看状态集合的定义：(i, j) 为前 1...i 元素划分成 j 份连续子数组的所有方案，f(i, j) 为这些所有方案的最优解(即最大平均值和)

然后再看如何转移：
1. j = 1 时，即划分为 1 份，此时直接利用定义计算平均值，我们可以利用前缀和数组 `prefix[]` 来简化计算平均值
2. 当 j > 1 时，我们可以将区间 [1, i] 划分为 [1, k] 与 [k + 1, i]。 其中 k 应该属于 `[1, i)` 与 `[j - 1, +∞)` 的交集，即 [1, k] 为前 j - 1 个数的划分，而 `[k + 1, i]` 为最后一个子数组单独划分出来。

这题的边界条件的转移状况是最难想象的：

代码：
```c++
double largestSumOfAverages(vector<int>& nums, int k) {
    int n = nums.size();
    double s[n + 1]; s[0] = 0;
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];       
    vector<vector<double> > f(n + 1, vector<double>(k + 1, 0));
    for (int i = 1; i <= n; i ++ ) f[i][1] = s[i] / i;      
    for (int i = 1; i <= n; i++) {
        for (int j = 2; j <= k && j <= i; j++) {
            for (int x = 2; x <= i; x ++ ) {
                f[i][j] = max(f[i][j], f[x - 1][j - 1] + (s[i] - s[x - 1]) / (i - x + 1));
            }
        }
    }
    return f[n][k];
}
```

## 823. 带因子的二叉树
定义：用 `f(i)` 表示以 `arr[i]` 为根节点的二叉树的个数，最终答案即为 `Σf(i)` `其中 i 属于 [0 ... n - 1]` 。

那么状态转移方程分析：

实际上这是一个 `斐波那契数列问题`，就像 `最长递增子列一样`，我们通过 map 来加速寻找合法的子树，而此题的 `合法性判断` 与 `两个键的键盘` 如出一辙。唯一不同的是不能用 根号来加速。因为这里`不是枚举数字`，而是`枚举下标`，所以一定是 O(n^2) 而不是 O(n*√n)

我们可以枚举 arr 中的每一个数 a 作为二叉树的根节点(根节点一定最大)，然后枚举枚举左子树的值 b，若 a 能被 b 整除，则右子树的值为 a / b，若 a / b 也在arr 中，则可以构成一棵二叉树。此时，以 a 为根节点的二叉树的个数为 `f(a) = f(b) × f(a / b)`，其中 f(b) 和 f(a / b) 分别为左子树和右子树的二叉树个数。

具体的，我们利用哈希表来记录 arr 内的每一个数，用来加速判断 arr 数组中是否当 a 存在时有`与之相符`的右子树 b 也存在 arr 中。

注意：`只有一个节点的二叉树`也被算作符合题意的二叉树

代码如下：

```c++
int numFactoredBinaryTrees(vector<int>& arr) {
    sort(arr.begin(), arr.end());
    unordered_map<int, int> idx;
    int n = arr.size();
    for (int i = 0; i < n; i ++ ) idx[arr[i]] = i;
    vector<LL> f(n, 1);
    for (int i = 0; i < n; i ++ ) 
        for (int j = 0; j < i; j ++ ) 
            if (arr[i] % arr[j] == 0 && idx.count(arr[i] / arr[j]))
                f[i] = (f[i] + f[j] * f[idx[arr[i] / arr[j]]]) % MOD;
    LL ans = 0;
    for (int i = 0; i < n; i ++ )
        ans = (ans + f[i]) % MOD;
    return ans;
}
```

## 837. 新 21 点
此题亦与 `688. 骑士在棋盘上的概率` 和 `576. 出界的路径数` 的思想路线相同，但是`不同的正序或逆序遍历方式`会导致时间复杂度大不相同。

在上面两题中记忆化搜索明显更好理解。

首先进行 记忆化搜索，定义 DFS(i) 为当前得分为 i 的情况下，我得分不超过 n 的概率。

代码如下：
```c++
vector<double> memo;
int n, k, maxPts;
double DFS(int i) {
    if (i > n) return 0;            // 如果得分 > n 则概率为 0
    if (i >= k) return 1;           // 如果得分 <= n 且 >= k 则返回 1.
    if (memo[i] != -1) return memo[i];

    double &res = memo[i]; res = 0;
    for (int j = 1; j <= maxPts; j ++ ) {       // 进行概率方程的状态转移。
        res += DFS(i + j) / maxPts;
    }
    
    return res;
}
double new21Game(int _n, int _k, int _maxPts) {
    n = _n, k = _k, maxPts = _maxPts;       // 特别注意，将变量要放在前面赋值，否则会导致后面的 memo 赋值 n + 1 大小时出错！！！
    memo = vector<double>(n + 1, -1);
    return DFS(0);
}
```

`进行数学优化`，注意，无论是数学优化还是空间压缩，我们都必须将记忆化搜索改为递推形式之后才能进行。因为在记忆化搜索的代码上采用状态压缩和数学优化令人头晕，在递推形式上进行空间压缩和数学优化令人看得清。

定义状态：定义 `dp[x]` 为她`当前得分`为 x 时，能获胜(即`k <= 最终分数 final <= n`)的概率。

首先，我们初始化状态集合：f(k ... k + maxPts - 1) ：因为得分达到到 k 以上就不会再增加了，所以最后一次增加只能处于得分为 k - 1 处，那么不再增加得分的分界线就在 `[k, +∞)` 而最后一次最多只能增加得分 `maxPts` 。所以我们可以精确地得知 `不再增加得分的区间为:` `[k, k + maxPts - 1]`。显然，在这个区间内，小于等于 n 的概率为 1， 大于 n 的概率为 0。

那么状态转移方程为：`f(x) = Σ{f(x + i)} / maxPts` `其中 i 属于 [1 ... maxPts]，x 的遍历从 k - 1 -> 0` 这样的`正序遍历方法`时间复杂度为 O(n + k * maxPts) 其中主要是最后一个 `k * maxPts 占大头`

如何优化时间复杂度？直接利用数学方法。此时需要`逆序遍历`。维护一个变量 s 代表 Σf(x + i) 那么每次不必直接利用遍历 `[i ... maxPts]` 来算出 s，再算计算出 `f(i)` 而是让 `f(i) = s / maxPts`。再利用 `s = s - f(i + maxPts) + f(i)` 来维护 s 的更新即可用 O(k + maxPts) 的线性时间计算出答案。


```c++
double new21Game(int n, int k, int maxPts) {
    vector<double> f(k + maxPts, 0);
    double s = 0;
    for (int i = k; i < k + maxPts; i ++ )
        f[i] = i <= n ? 1 : 0, s += f[i];
    for (int i = k - 1; i >= 0; i -- )
        f[i] = s / maxPts, s = s - f[i + maxPts] + f[i];
    return f[0];
}
```

## 838. 推多米诺

方法 1： BFS 模拟推多米诺过程。

分析：有多个点值得注意：
1. 以时刻为赋值的关键，每个`当前时刻`对一组`下一时刻`将要倒下的牌进行更新
2. 每个位置的牌仅受 `上一时刻的相邻的牌的倒向` 影响，且只能更新一次。 
3. 如果在`同一时刻被推两次`，则说明该处应该竖直立正

体会：又一次`将 v[] 数组的巧妙使用推向了高峰！`

算法模拟如下：

```c++
string pushDominoes(string s) {
    int n = s.size();
    vector<int> v(n, 0);
    queue<int> q;
    for (int i = 0; i < n; i ++ )
        if (s[i] != '.') q.push(i), v[i] = 1;
    int dis = 1;
    while (q.size()) {
        dis ++ ;
        for (int i = q.size(); i; i -- ) {
            int x = q.front(); q.pop();
            int nex = s[x] == 'L' ? x - 1 : x + 1;                  
            if (nex < 0 || nex >= n || s[x] == '.') continue;       // 如果当前是 '.' 或者不符合题意，则
            if (!v[nex]) v[nex] = dis, q.push(nex), s[nex] = s[x];  // 如果从没有被推过，则推他，并将 s[nex] 赋值为 s[x]
            else if (v[nex] == dis) s[nex] = '.';           // 如果在 `同一时刻被推两次` ，则说明该处应该竖直立正.
        }
    }
    return s;
}
```

模拟 + 双指针算法：

```c++
string pushDominoes(string s) {
    s = 'L' + s;
    string res = s;
    int i = 0, n = s.size();
    char left = 'L';
    while (i < n) {
        int j = i + 1;
        while (j < n && s[j] == '.') j ++ ;
        if (j >= n) {
            if (left == 'R')
                for (int k = i; k < n; k ++ ) res[k] = 'R';
            break;
        }
        if (left == 'L' && s[j] == 'L')
            for (int k = i; k <= j; k ++ ) res[k] = 'L';
        else if (left == 'L' && s[j] == 'R')
            int doNothing;
        else if (left == 'R' && s[j] == 'L') 
            for (int k = i, t = j; k < t; k ++ , t -- )
                res[k] = 'R', res[t] = 'L';
        else if (left == 'R' && s[j] == 'R')
            for (int k = i; k <= j; k ++ )
                res[k] = 'R';
        i = j, left = s[j];
    }
    return res.substr(1);
}
```

## 845. 数组中的最长山脉
此题与`764. 最大加号标志`，与`221. 最大正方形`一样，是一个道 `几何形状 DP 问题`

我们将 `l[i]` 定义为以 i 点为起点，向`左边`能延申出的最长递减`子数组(不是子序列)`。

同理，`r[i]` 定义为以 i 点为起点，向`右边`能延申出的最长递减`子数组(不是子序列)`。

那么由容斥原理得出，以 i 为中间节点的山脉子数组长度为：`l[i] + r[i] - 1` (前提是左右两边最大延伸长度都大于等于 2)。
```c++
int longestMountain(vector<int>& arr) {
    int n = arr.size();
    vector<int> l(n, 1), r(n, 1);
    for (int i = 1; i < n; i ++ )
        if (arr[i - 1] < arr[i]) l[i] = l[i - 1] + 1;
    for (int i = n - 2; i >= 0; i -- )
        if (arr[i] > arr[i + 1]) r[i] = r[i + 1] + 1;
    int ans = 0;
    for (int i = 0; i < n; i ++ )
        if (l[i] >= 2 && r[i] >= 2)
            ans = max(ans, l[i] + r[i] - 1);
    return ans;
}

```
## 55. 跳跃游戏
此题是一个`模拟 + 贪心`题，模拟方法需要记忆，代码很巧妙：我们只需要记录下`在 i 处时，我们能够到达的最大位置的下标位置 cur = max(cur, i + nums[i])`即可，如果当前位置在 i 处更新 cur 之后，cur == i 即不能再往前走了，就代表无法继续前进，即不能到达终点。能够到达终点的退出条件为：`cur < n - 1 && i < n` 因为当`cur >= n - 1 时`就已经代表了我能够到达的最大位置为 `n - 1`，即可以到达最终点.

```c++
bool canJump(vector<int>& nums) {
    int cur = 0, n = nums.size();
    if (n == 1) return 1;
    for (int i = 0; i < n && cur < n - 1; i ++ ) {
        cur = max(cur, i + nums[i]);
        if (cur == i) return 0;
    }
    return 1;
}
```
更好的模拟方法介绍：这种模拟方法适用范围不广，但是就其代码而言更简洁，但普适性不强，不推荐。
```c++
bool canJump(vector<int>& nums) {
    int maxL = 0, n = nums.size();
    for (int i = 0; i < n; i ++ ) 
        if (maxL < i) return 0;
        else maxL = max(maxL, nums[i] + i);
    return 1;
}
```

## 45. 跳跃游戏 II
此题贪心策略：考虑下一次跳跃能够达到的最远距离，即枚举当前点 i 处的跳跃时，可跳越出去的位置，并选取`对应位置距离的那个点 nei`，即让 `i = nei`，其中 nei = i + j。

时间复杂度为：O(n^2)

细节：为了防止 i + j 溢出 nums 数组，必须特判 i + j >= n - 1 即 nei 的位置。即分情况讨论：
1. 如果能直接跳到最后一个位置，则直接返回前面累计的 ans 并加上 1 即可： `ans + 1`
2. 如果不能一次性跳到最后一个位置，则去选取下一次能挑到更远的位置的那个对应的 j ，并让 nei = i + j。
```c++
int jump(vector<int>& nums) {
    int cur = 0, ans = 0, n = nums.size();
    for (int i = 0; i < n - 1; ) {
        int nei = i;
        for (int j = nums[i]; j; j -- ) // 枚举此次跳跃时，可跳越出去的位置
            if (i + j >= n - 1) // 如果能直接跳到最后一个位置
                return ++ ans;
            else if (i + j + nums[i + j] > cur) // 如果不能一次性跳到最后一个位置，选取下一次能挑到更远的位置的那个对应的 j，有点绕。
                cur = i + j + nums[i + j], nei = i + j; 
        ans ++ , i = nei;
    }
    return ans;
}
```


惰性更新写法：O(n)：

我们不去寻找我们下一次要在哪个点跳跃，而是`在需要跳跃的时候`更新一下`如果可以更新`的话，最远能到的距离 curR, 和 ans。

举一反三：就像 `871. 最低加油次数` 问题一样，我们只有当油不够时，才拿出最大的那桶油加入邮箱。然后进行更新需要加油的次数。这就是惰性更新
```c++
int jump(vector<int>& nums) {
    int ans = 0, curR = 0, curMax = 0, n = nums.size();
    for (int i = 0; i < n - 1; i ++ ) {     // 
        curMax = max(curMax, nums[i] + i);
        if (i == curR) ans ++ , curR = curMax;
    }
    return ans;
}
```


## 45. 跳跃游戏 II

## 871. 最低加油次数
这是一道优先队列贪心问题，如同 `45. 跳跃游戏 II` 一样，它们的基本策略思想都是贪心去选择一个最大值，不同之处在于：此题是选择`之前所有`可以选择的最大值贪心，而`45. 跳跃游戏 II`是去选择在我在当`前处 i 往后`的下一次跳跃能到达的最大值贪心。所以代码和思路也有小不同，但整体的贪心思想是相同的。

算法模拟如下：

记 remain 为当前油量能够跑出的公里数，pos 为当前位置，i 为`当前位置是 pos 时，已经经过的加油站(包括此时这个位置)`，q 为可用加油站(大根堆)，其堆顶为可用加油站中的最大量加油站

当 pos < target 时，我们进行循环更新 pos：
1. 当前位置没有油时，我们进行加油的模拟，取出堆中最大的油量，并加入。`如果堆为空`，说明所有的油都加过，但是却无法移动到下一个加油站或无法到达终点了。
2. 加完油后，我们将对 pos 和 remain 进行更新，对 pos 来说，更新其最远位置，对于 remain 来说，进行置 0。
3. 并对 heap 中的可用加油站进行更新，加入 pos 之前的所有加油站。

```c++
int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations) {
    priority_queue<int> q;
    int n = stations.size(), pos = 0, i = 0, remain = startFuel, ans = 0;
    while (pos < target) {
        if (remain == 0)    // 
            if (q.size()) // 当前位置没有油时，我们进行加油的模拟，取出堆中最大的油量，并加入
                remain += q.top(), q.pop(), ans ++ ;  
            else    // `如果堆为空`，说明之前所有的油都加过，但是却无法移动到下一个加油站或无法到达终点了。
                return -1;
        pos += remain, remain = 0;  // 对位置 pos 和 remain 进行更新
        while (i < n && stations[i][0] <= pos) // 对 heap 进行更新
            q.push(stations[i ++ ][1]);
    }
    return ans;
}

```

## 873. 最长的斐波那契子序列的长度

此题的状态定义需要学习，直接上定义：

定义：`f(i, j)︰表示以 A[i], A[j] 为结尾的斐波那契数列的最大长度，即此时序列的样子为：( ..., A[i], A[j])`

状态转移：由于 A[i], A[j] 为该序列的最后两个数，则我们应该找到一个数 x ，使得满足` x + A[i] == A[j]`。

所以状态转移方程为： `f(i, j) = f(k, i) + 1, (其中 k 满足 k + A[i] == A[j])`

注意我们只能保留长度 >= 3 的子序列，所以需要特殊处理：`f(i, j) = max{f(k, i) + 1, 3}` 因为只要找到了这样一个 k(即找到了`上面分析的 x 对应的下标`)，则注定长度 >= 3，所以用这个代码就解决了，即 f(k, i) == 0 时，也能让 f(i, j) = 3.

细节：
1. 我们用 mp[] 来记录每个值的坐标，以便于用 O(1) 的时间去找到正确的 x ，而不是 O(n) 的时间复杂度。
2. 非法的`伪`斐波那契数列判断：斐波那契数列一定需要是递增序列，否则`即使`满足 a + b = c 但是`如果`有 a > b ，那么此数列仍然`不是`斐波那契数列。
2. 为了避免让斐波那契数列出现非递增的情况：例：`A[] = [2,5,7,10,17]`时，我们要明确一个点，`f(2, 4)` 应该等于 0，因为 以 7, 17 作为结尾的子序列： `10, 7, 17` 不应该成为一个合法数列，但是如果`只按`以上的分析来看，这就是一个合法的数列。所以我们为了避免这种情况发生，在枚举 A[i] 时，我们必须`从大到小`进行枚举。只允许枚举到 A[j] / 2 这样一个大小，`即 A[i] > A[j] * 2, 否则非法`。

直接上代码：

```c++
int lenLongestFibSubseq(vector<int>& arr) {
    unordered_map<int, int> mp;
    int n = arr.size(), ans = 0;
    for (int i = 0; i < n; i ++ ) mp[arr[i]] = i; // 记录下所有的 x，以便于找到对应的 x，并得到对应的下标 k
    vector<vector<int> > f(n, vector<int>(n));  // 初始化 f(i, j) 全部为 0
    for (int j = 2; j < n; j ++ )
        for (int i = j - 1; i; i -- ) {         // 我们为了避免非递增的 伪斐波那契数列 ，我们必须从大到小进行枚举
            if (arr[i] * 2 <= arr[j]) break;    // 根据上面分析的最后面一条，如果不是递增序列则不合法，则剪枝
            if (!mp.count(arr[j] - arr[i])) continue;   // 这里直接去判断是否存在这样的 x，如果不存在则直接继续找
            int k = mp[arr[j] - arr[i]];        // 如果存在则进行状态转移
            f[i][j] = max(3, f[k][i] + 1);      
            ans = max(ans, f[i][j]);
        }
    return ans;
}
```

总结：此题的寻找对应的转移集合的方式思想和 `前缀和哈希`，`823. 带因子的二叉树` 相同，都是`利用哈希表`去寻找能够转移来的出处。好好去体会一下。

## 877. 石子游戏  区间DP + 博弈论DP
根据之前写过的的各种`预测赢家`，`石子游戏`等题目，我们利用同样的套路：

定义 f(i, j) 为当前玩家只剩下区间 [i, j] 时能得到的与另一玩家的`最优(最有利于自己)的差值`.

那么状态转义方程为：`f(i, j) = max{nums[i] - f(i + 1, j), nums[j] - f(i, j - 1)}`

```c++
bool stoneGame(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int> > f(n, vector<int>(n));
    for (int i = 0; i < n; i ++ ) f[i][i] = nums[i];
    for (int i = n - 2; i >= 0; i -- )
        for (int j = i + 1; j < n; j ++ )
            f[i][j] = max(nums[i] - f[i + 1][j], nums[j] - f[i][j - 1]);
    return f[0][n - 1] > 0; 
}

```

## 鸡蛋掉落
不同的状态集合定义会导致不同的状态转移方程，这里我们首先介绍官方解法3，一种更好的状态定义思路。

定义：f(k, m) 为我们有 k 个鸡蛋，可以操作 m 次，此时我们能够确定出的最大楼层高度。

边界条件：由物理实际含义得知：
1. 当 k == 1 时，如果操作 m 次，则我们可以确定 m 楼，因为当鸡蛋只有一个时，只能从第一层往上一个一个试(否则无法正确判断)，所以试了 m 次后能够确定出 m 楼
2. 当 m == 1 时，无论鸡蛋 k 有多少，都只能确定 1 个楼层，因为我们从 1 楼往下扔，只能确定 0 层不碎，或者 1 层不碎。(层数从 0 ~ n 即有 n + 1 层)

那么状态转移方程为：`f(k, m) = f(k - 1, m - 1) + f(k, m - 1) + 1` 解释：首先我们`随意`挑选一个楼层进行测试，因为它这个题目含义是要保证最差情况，所以`随机挑选我们把它理解为就是我们选中了那个最差的那种扔鸡蛋的情况`。
1. 在当前楼层摔碎了则在下边的楼层继续测试：`f(k - 1, m - 1)`
2. 在当前楼层没有摔碎则在上方的楼层继续测试：`f(k, m - 1)`
3. 而当前楼层可以知道它是不是 <= 临界楼层所以加上这个楼层` + 1`

那么我们枚举操作次数，当 f(k, m) < n 时，我们不断增加操作次数 m，直至 f(k, m) 可以确定 n 层楼，那么最小操作次数 m 就是那个答案。

可以加入 memo 来进行记忆化搜索。

代码如下：

```c++
vector<vector<int> > memo;
int f(int k, int m) {
    if (k == 1 || m == 1) return m;
    if (memo[k][m] != -1) return memo[k][m];
    int &ans = memo[k][m]; ans = f(k - 1, m - 1) + f(k, m - 1) + 1;
    return ans;
}
int superEggDrop(int k, int n) {
    int m = 1;
    memo = vector<vector<int> >(k + 1, vector<int>(n + 1, -1));
    while (f(k, m) < n) m ++ ;
    return m;
}
```

## 894. 所有可能的真二叉树
我们直接利用递归来做：注意这种二叉树也叫满二叉树(国际叫法)，和 408 教材的所谓的满二叉树不同。

易知：
1. 如果是偶数，则不可能为这种满二叉树形状
2. 如果 n == 1 则直接构造一个只有一个根节点的树
3. 如果是奇数且 n > 1，则进行遍历构造，即令左子树的数量为 cntL, 那么计算出右子树的数量 cntR = n - 1 - cntL。

那么我们可以递归得到左子树数量为 cntL 时的树集合列表，同时递归得到右子树数量为 cntR 时的树集合列表。那么我们就通过这些树来构造当前 n = cntL + cntR + 1 的树的集合，其构造转移方程可写为：
```c++
for (auto e1 : left)
    for (auto e2 : right) {
        TreeNode *root = new TreeNode(0);
        root->left = e1;
        root->right = e2;
        res.push_back(root);
    }
```
那么直接写代码：
```c++
unordered_map<int, vector<TreeNode*> > memo;
vector<TreeNode*> allPossibleFBT(int n) {
    if (memo.count(n)) return memo[n];
    vector<TreeNode*> &res = memo[n];
    if (n & 1 == 0) return res;     // 如果是偶数，则不可能为这种满二叉树形状
    if (n == 1) { res.push_back(new TreeNode(0)); return res; } // 如果 n == 1 则直接构造一个只有一个根节点的树
    for (int cntL = 1; cntL <= n - 2; cntL += 2) {  // 进行遍历构造，即令左子树的数量为 cntL, 来对左子树进行递归构造，同样只能是奇数才有意义
        int cntR = n - 1 - cntL;        // 计算出右子树的数量 cntR = n - 1 - cntL
        vector<TreeNode*> left = allPossibleFBT(cntL);  // 递归得到左子树数量为 cntL 时的树集合列表
        vector<TreeNode*> right = allPossibleFBT(cntR); // 递归得到右子树数量为 cntR 时的树集合列表

        for (auto e1 : left)    // 通过两个子树集合列表来构造当前的树的集合列表
            for (auto e2 : right) {
                TreeNode *root = new TreeNode(0);
                root->left = e1;
                root->right = e2;
                res.push_back(root);
            }
    }
    return res;
}
```


## 2411. 按位或最大的最小子数组长度

我们通过此题来介绍一个关于 `按位或 + 连续子数组 + 区间性质` 属性类型题的一个通用模板：
首先介绍 ors 数组，其签名为：`typedef pair<int, int> PII; vector<PII> ors; ` 实际上 ors 本质是一个`动态规划经过维度压缩过后的数组`，本来他的签名应该是这样：`vector<vector<PII> > ors` ，`ors[i][j].first`代表了以 i 为起始节点，从左往右`到数组最后一个元素`能够算出的子数组的不同的 `按位或的值`，即`有 j 个不同的按位或的值`，但是我们可以隐藏第一个维度 i，让 `ors[j].first` 代表`**当前** i 为起始节点`的不同的按位或的值。而 ors[j].second 代表这个值所在的连续区间的左端点，因为对于一个按位或的值，可以存在一整个区间，让其从 i 到该区间任何下标点的按位或的值相同。举例：
```
{... i ... [x, x, x], [y, y], [z, z, z, z, z]}
```
可以看到，在第一个连续区间 `[x, x, x]` 中，从 i 到这个区间(大小为 3 )的任何一个下标点的按位或值相同，值为 x ，然后在第二个区间(大小为 2) [y, y] 也是同理。由或运算只增不减原理得知：`x < y < z` 。所以我们只记录下`这些连续区间代表的不同的按位或的值`，`以及这个区间对应的最左端的下标点`即可。根据数据范围，我们`最多拥有 30 个不同的按位或的值`，所以`最多有 30 个这样的连续区间`。然后我们`根据数学的迭代性质`(因为后续的对应的 ors 可以通过上一个迭代出的 ors 得到)来`省略动态规划的第一个维度 i`，直接让 ors 代表当前节点的对应数组。

`特别注意`：在代码实现中，ors 数组是`从或值最大到或值最小`来存储的，这是因为我们每次都用 ors.push_back({0, i}) 来`将最小的或值插入`到 ors 的最后。所以，ors 是按或值递减的顺序排列的，而不是由小到大。即先是存储 z，然后是 y，然后是 x ... 

那么我们就先研究代码，在代码的注释中来看具体在代码中是如何实现上述思想的：

```c++
vector<int> smallestSubarrays(vector<int> &nums) {
    int n = nums.size();
    vector<int> ans(n);
    vector<pair<int, int>> ors; // 按位或的值 + 对应子数组的右端点的最小值
    for (int i = n - 1; i >= 0; --i) {  // 由于我们的 ors[] 数组是从左到右的，所以我们必须反向迭代，即 i 从右到左遍历
        ors.push_back({0, i});      // 这里可知，或值越小，其越加靠后，且下标 i 越小，即 second 越小
        int k = 0; ors[0].first |= nums[i];      // 利用双指针进行 -> 删除有序数组中的重复项。相当于 c++ 库中的 unique() 函数
        for (int j = 1; j < ors.size(); j ++ ) {    // 特别注意
            ors[j].first |= nums[i];                // 如果我们将这一步拆开来看作两个循环，即 更新或值 和 去重 这两个循环拆开来看的话，可能更清晰，见下一个代码段。
            if (ors[k].first == ors[j].first)   // 如果 `或值` 相同，则下标选小的，由 ors 的 push_back() 操作可知，j 越靠后，其对应的下标 i 越小，所以一定是更新成最靠后的 j 的下标 second
                ors[k].second = ors[j].second; // 合并相同值，下标取最小的
            else ors[ ++ k] = ors[j];           // 如果 `或值` 不同，则加入新数组，注意是 `原地` 构造新数组
        }
        ors.resize(k + 1);
        // 本题只用到了 ors[0]，如果题目改成任意给定数字，可以在 ors 中查找
        ans[i] = ors[0].second - i + 1;
    }
    return ans;
}
```

我们将 `更新或值` 和 `去重` 这两个循环拆开来看的话，可能更清晰，见此代码段。
```c++
vector<int> smallestSubarrays(vector<int> &nums) {
    int n = nums.size();
    vector<int> ans(n);
    vector<pair<int, int>> ors; // 按位或的值 + 对应子数组的右端点的最小值
    for (int i = n - 1; i >= 0; i -- ) {  // 由于我们的 ors[] 数组是从左到右的，所以我们必须反向迭代，即 i 从右到左遍历
        ors.push_back({0, i});  
        for (int j = 0; j < ors.size(); j ++ ) ors[j].first |= nums[i];     // 先更新第一个维度 first，然后再进行去重，即拆成了两部分
        int k = 0;  
        for (int j = 1; j < ors.size(); j ++ ) 
            if (ors[k].first == ors[j].first)   // 如果 `或值` 相同，则合并相同值，下标取最小的，即后面遍历的 second 的值
                ors[k].second = ors[j].second;
            else ors[ ++ k] = ors[j];           // 如果 `或值` 不同，则加入新数组，注意是 `原地` 添加入新数组
        
        ors.resize(k + 1);
        // 本题只用到了 ors[0]，如果题目改成任意给定数字，可以在 ors 中查找
        ans[i] = ors[0].second - i + 1;
    }
    return ans;
}
```

## 898. 子数组按位或操作
我们直接利用上面的模板来做这道题：只需要利用 unordered_set 来去重这个连续子数组的不同值即可

```c++
int subarrayBitwiseORs(vector<int>& nums) {
    vector<PII> ors;
    int n = nums.size();
    unordered_set<int> S;
    for (int i = n - 1; i >= 0; i -- ) {
        ors.push_back({0, i});
        for (int j = 0; j < ors.size(); j ++ ) ors[j].first |= nums[i]; // 先对 ors[] 数组进行更新
        int k = 0;
        for (int j = 1; j < ors.size(); j ++ )
            if (ors[k].first == ors[j].first) ors[k].second = ors[j].second;
            else ors[ ++ k] = ors[j];
        ors.resize(k + 1);
        for (int j = 0; j < ors.size(); j ++ )
            S.insert(ors[j].first);
    }
    return S.size();
}
```

## 902. 最大为 N 的数字组合

此题直接可略去 special 参数，因为`此题要求我们只需关系构造出来的 t 是否是数字`，即只需要 isNum 即可，而`无需关心`这个 t 有什么需要符合题意对应的属性 special

```c++
string s;
int len;
vector<int> memo;
vector<string> digits;
int f(int i, bool isLimit, bool isNum){ // 略去了 special 参数
    if (i == len) return isNum;
    if (!isLimit && isNum && memo[i] != -1) return memo[i];
    int res = 0;
    if (!isNum) res = f(i + 1, 0, 0);
    int up = isLimit ? s[i] : '9';
    for (auto d : digits) {
        if (d[0] > up) break;
        res += f(i + 1, isLimit && d[0] == up, 1);
    }
    if (!isLimit && isNum) memo[i] = res;
    return res;
        
}
int atMostNGivenDigitSet(vector<string>& _digits, int n) {
    s = to_string(n); digits = _digits; len = s.size();
    memo = vector<int>(len, -1);
    return f(0, 1, 0);
}
```

## 918. 环形子数组的最大和
解法一：注：解法 1.1 与 解法 1.2 利用`数学反向思考`，用两个不同的基本解法来分为两种解法 1.1 与 1.2 。

解法 1.1：`数学反向思考 + 动态规划` 本解法基于的基本解法为`动态规划`，需要特判一种情况：

这个方法具有技巧性，`直接去看灵神的题解`，或者`外国的那个题解`.详细的说明为什么要求 `最小子数组和` 与 `最大子数组和`，并通过这两个来构造最终的答案。但是答案中有一个说法错了，那个特殊情况应该是`数组元素全为负数`，而不是最小子数组是整个数组，即那个图的最后一个例子是不对的。但是灵神的代码是对的，因为在数学上，这个数学逻辑和这种代码是等价的。

解法 1.2：`数学反向思考 + 前缀和` 本解法基于的基本解法为 `前缀和` 解法，由于没有基于动态规划，所以从数学上可以证明，不需要去特判特殊情况，可以直接返回答案。如何`用前缀和解决前置问题`，在解法二中有讲。

本解法的网址：

https://leetcode.cn/problems/maximum-sum-circular-subarray/solutions/2351138/python3javacgotypescript-yi-ti-yi-jie-we-uug0/

本解法的代码：
```c++

int maxSubarraySumCircular(vector<int>& nums) {
    const int INF = 0x3f3f3f3f;
    int pmi = 0, pmx = -INF;    // pmi 为前缀和最小值，pmx 为前缀和最大值
    int ans = -INF, s = 0, smi = INF;   // s 为当前前缀和，ans 为普通的最大子数组和，smi 为最小子数组和。
    for (int x : nums) {
        s += x;
        ans = max(ans, s - pmi);
        smi = min(smi, s - pmx);
        pmi = min(pmi, s);
        pmx = max(pmx, s);
    }
    return max(ans, s - smi);
}

```

解法二：`前缀和 + 单调队列 + 动态规划`，此方法直接让我们抽象出一个`通用解法`：在一个长度为 n 的数组 nums[] 上，寻找长度不超过 k 的最大子数组和。

本解法`基于前缀和`基本解法，但是`相当于增加了`滑动窗口的概念。并且我们需要找到这个窗口内的最小值，所以维护一个单调递增的队列。

首先我们来思考，如何使用前缀和来解决前置问题：`53. 最大子数组和`：

维护一个最小前缀值，然后遍历前缀和，注意初始化 `min_pre = 0, ans = -INF` 即开始让最小前缀和 min_pre 为 0，即代表实际意义为：最开始没有加入任何元素的前缀和，然后 ans = -INF，即还未遍历时，让 ans 为负无穷，后面会更新到正确的值。

`特别值得注意`的数学上的一件事：min_pre 的这个元素不会被包含在目标子数组内！！！因为根据数学定义，他被减去了。
```c++
const int INF = 0x3f3f3f3f;
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> s(n + 1);
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];
    int min_pre = 0, ans = -INF;    // 
    for (int i = 1; i <= n; i ++ ) {
        ans = max(ans, s[i] - min_pre);     // 值得注意的是，min_pre 这个当前这个点不会被包含在子数组内。
        min_pre = min(min_pre, s[i]);
    }
    return ans;
}
```

然后通过这个基础，我们需要`严格的区分前缀和与对应子数组的边界条件`来正确地使用单调队列来处理此问题：在一个长度为 n 的数组 nums[] 上，寻找长度不超过 k 的最大子数组和：

1. `最首要`的事：我们的`不是从当前构造的这个单调队列`来选取窗口内的`最小前缀和 s[q.front()]`。而是在`前一个`构造的窗口内去选取，这是因为我们让窗口大小为 k，举个例子：
```
{..., [1, 2, ... , k], i, ... }
```
可以看到，当前遍历到了 i，那么可知道`前 k 个数`中最小的前缀和，即上一个窗口，而该区间最左端的端点为 i - k，那么如果恰好这个点的前缀和最小的话，则有 `s[i] - s[i - k]` 正好有 `i - (i - k) = k` 个数，即正好是符合题意，即此时子数组为：`[2, 3, ..., k, i]` 即最右边的数没有被包括在子数组内。

2. 一个关键点：在单调队列模板的`何处`更新答案是`关键`。我们需要明确：应该在`还未形成当前的单调队列窗口`时进行更新。因为由 1 中分析出，我们应该在上一次的单调队列中进行更新

`总结`：所以如何利用单调队列，比单调队列模板重要，这个的`创新点`就在于是前一个单调队列进行更新，而不是当前的单调队列进行更新。

```c++
int maxSubarraySumCircular(vector<int>& nums) {
    nums.insert(nums.end(), nums.begin(), nums.end());  // 构造出一个重复两次的数组
    int n = nums.size(), k = n / 2;     // 计算出对应的 n 与 k。
    vector<int> s(n + 1);
    for (int i = 1; i <= n; i ++ ) s[i] = nums[i - 1] + s[i - 1];
    deque<int> q; q.push_back(0);   // 首先将 s[0] = 0 ，即下标 0 入队，进行初始化，就如同前置问题中，让 min_pre 初始化为 0.
    int ans = -INF;     // 就像前置问题中，让 ans = -INF。
    for (int i = 1; i <= n; i ++ ) {
        while (q.size() && q.front() < i - k) q.pop_front();    // 此时代表的 前一次 构造出的单调队列
        ans = max(ans, s[i] - s[q.front()]);        // 在 还未形成当前的单调队列窗口 时进行更新，即在前一次的单调队列进行更新答案。
        while (q.size() && s[q.back()] >= s[i]) q.pop_back();   // 当前以及下一行代码是
        q.push_back(i);
    }
    return ans;
}
```



## 926. 将字符串翻转到单调递增
解法一：`贪心数学规律 + 前缀和 + 枚举`

暴力：如果暴力都不会，还写个屁的
这就像中心扩散思想，只不过 扩散过程从 O(n) 被 前缀和优化到了 O(1)，如果暴力中心扩散的话，就是 O

对于每一个位置 i，我们考虑：将其`左侧所有的数字都转化为 0`，将其`右侧的所有数字都转化为 1`，而`当前第 i 个数字`不转化，这样，我们就用这种`贪心`的数学规律，来得到对当前位置 i 来说，最少需要多少次操作，得到其左侧全 0， 右侧全 1 的情形。然后我们枚举每一个位置 i，迭代出最小的 ans。

代码实现上述思想：
1. 朴素实现：如果要使左侧所有的数字化为 0，右侧所有数字都化为 1，那么我们可以利用`经典的中心扩展暴力思想`来做，时间复杂度为 O(n) 。然后枚举每一个位置要求 O(n) 的复杂度，那么总时间复杂度为 O(n^2)
2. 前缀和加速实现：我们可以利用`前缀和思想`来代替`暴力中心扩展`法，使得这个步骤的时间复杂度为 O(1)。具体来说，我们记录下每个点的前缀和，然后就可以通过 `s[i - 1] 得到左侧 1 的个数`, 通过 n - i - (s[n] - s[i]) 得到右侧 0 的个数，从而得到转化所需的反转次数：`s[i - 1] + n - i - (s[n] - s[i])`。

前缀和实现：
```c++
const int INF = 0x3f3f3f3f;
int minFlipsMonoIncr(string str) {
    int n = str.size();
    vector<int> s(n + 1);
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + str[i - 1] - '0';
    int ans = INF;
    for (int i = 1; i <= n; i ++ ) 
        ans = min(ans, s[i - 1] + n - i - (s[n] - s[i]));
    return ans;
}
```
解法二：动态规划
就像`790. 多米诺和托米诺平铺` 和 `最长 zigzag 子序列`，`978. 最长湍流子数组` 一样，可以用状态 DP 来解决，我们使用两个状态：其中

定义：用 `f[i][0]` 和 `f[i][1]` 分别表示下标 i 处的字符`变`为 0 和 1 的情况下使得 `s[0..i]` 单调递增的最小翻转次数。更详细的解释: 注意这个`变`字即如果`当前为 0`，那么我要`变为 0` 则操作次数为 0，即`不用操作`，而我要变为 1 则操作次数为 1；对于当前为 1 时同理。而 `f[i][0 or 1]` 是使得这个串 `s[0..i]` 满足单调递增的最小翻转次数，说明我赋值完 `f[i][0 or 1]` 后，一定是单调递增序列。

那么状态转移方程为：
1. `f[i][0] = s[i] == '0' ? f[i - 1][0] : f[i - 1][0] + 1` 即如果要将此翻转为 0 状态且满足递增，则`上一次一定必须是 0 的状态`，否则上一次为 1 的话就不满足递增，所以只能从 0 状态转移过来。而如果 s[i] 为 0 则`不用`翻转，即`不用 +1`，如果为 1 则`需要翻转`，`即 +1`
2. `f[i][1] = s[i] == '0' ? min(f[i - 1][0], f[i -1][1]) + 1 : min(f[i - 1][0], f[i -1][1])` 即如果要将此翻转为 0 状态且满足递增，那么无论上一个状态是 0 还是 1 都可以，因为都是满足递增的，选择最小的那个即可。而如果 s[i] 为 0 则`需要翻转`，`即 +1`，否则则`不用`翻转，即`不用 +1`

实际上要理解以上文字含义，举一个例即可，即最终状态一定是形如这样的：`[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]` 

代码如下，利用了空间优化：特别注意，我们在空间优化时，即隐去一个维度进行空间优化时，有两个方法，只有使用这两个方法，才能让转移方程正确的更新！
1. 法一：利用 `backup 备份` 来进行更新：即使用 backup 来记录上一次的状态 f 的结果，然后用 backup 来更新当前的 f 。`这个方法是万能的`
2. 法二：利用 `调整遍历顺序` 来进行更新，这个方法`不是万能`的，必须类似于背包问题那种特殊的情形下，即能通过状态矩阵的左右上下的关系来判断如何转移才能不至于使下一个状态进行转移更新时，`所要用到的旧状态`已经被`更新为新状态` 而出现的错误。而这种错误就是最典型常见的空间压缩的错误即：`代码没能正确的实现数学逻辑`，也就是说，`数学逻辑表达式与代码表达式具有不一致性`

```c++
const int INF = 0x3f3f3f3f;
int minFlipsMonoIncr(string str) {
    int n = str.size();
    int f0 = str[0] == '0' ? 0 : 1, f1 = str[0] == '1' ? 0 : 1;
    for (int i = 1; i < n; i ++ ) {
        int f0_tmp = f0, f1_tmp = f1;       // 这里：为了避免 `所要用到的旧状态`已经被`更新为新状态` 而出现的错误，即`数学逻辑表达式与代码表达式具有不一致性`，我们必须利用 backup 来记录上一次的状态
        f0 = str[i] == '0' ? f0_tmp : f0_tmp + 1;
        f1 = str[i] == '1' ? min(f0_tmp, f1_tmp) : min(f0_tmp, f1_tmp) + 1;
    }
    return min(f0, f1);
}
```

## 931. 下降路径最小和
这题太经典了，不用分析，直接上答案：
```c++
const int INF = 0x3f3f3f3f;
int minFallingPathSum(vector<vector<int>>& mat) {
    int n = mat.size();
    for (int i = 1; i < n; i ++ )
        for (int j = 0; j < n; j ++ ) {
            int tmp = mat[i][j];
            mat[i][j] += mat[i - 1][j];
            if (j + 1 < n) mat[i][j] = min(mat[i][j], tmp + mat[i - 1][j + 1]);
            if (j - 1 >= 0) mat[i][j] = min(mat[i][j], tmp + mat[i - 1][j - 1]);
        }
    int ans = INF;
    for (int j = 0; j < n; j ++ )
        ans = min(ans, mat[n - 1][j]);
    return ans;
}
```

## 940. 不同的子序列 II
解法一：字符集 + 动态规划
`错误解法`：令 f(i) 为前 1...i 中的所有不同子序列(注意：`考虑空序列也算一个子序列`，在最后返回答案时减去空序列即可)，状态转移方程为：f(i) = f(i - 1) + f(i - 1), 即将当前字符加入到上一次每个子序列的后面构成一个新的子序列，很遗憾，这种方法是错误的，因为他会导致重复，比如："aa"，这样就是 f(0) = 2 (空序列也算子序列，共有：`""` 与 `"a"` 两个), f(1) = 4 (共有`"", "a", "a", "aa" 四个`). 但是实际上，f(1) 应该是等于 3，因为这个 "a" 是重复算了一次。

正确解法：换个定义方式：令 f(i, c) 为前 1...i 中，以 c 字符为结尾的所有子序列(这种定义方法不需要考虑空子序列)。则状态转移方程为：

`f(i, c) = 1 + Σf(i - 1, c_j), 其中 j 从 0 到 25，即字符集大小 ` 其中 1 为当前这个字符 s[i], 而后面那个求和则是将 s[i] 加入到每个之前的子序列的后面构成新的子序列。

解法二：数学 + 动态规划

如何让解法一中的正确解法变为正确解法呢？利用数学来处理之前重复的元素数量 pre_repeat。

即正确的状态转移方程应该是： `f(i) = f(i - 1) * 2 - pre_repeat` 那么这个 pre_repeat 该如何计算呢？




## 115. 不同的子序列
这是关于子序列的模式匹配，还是`老套路`的定义状态集合：定义`f[i][j]` 为考虑` s 中 [0, i] 内的所有子序列`中，与 t 的`子串` t[0, j] 的匹配个数。

特别注意定义内，s[0 ... i] 是子序列，而 t[0 ... j] 是子串，两者是不一样的。

那么状态转移方程：`f[i][j] = s[i - 1] == t[j - 1] ? f[i - 1][j] + f[i - 1][j - 1] : f[i][j] = f[i - 1][j];` 含义为：

如果 s 的最后一个字符能匹配到 t 的最后一个字符，那么可以从两种状态转移过来：
1. 即`一定`用 `s[i]` 来进行匹配，也就是 t[0 .. j - 1] 与 s[0 ... i - 1] 的总匹配数量: `f[i - 1][j - 1]`   类似于曲线救国：`先`一定不去匹配，`然后`一定去匹配，因为 t 是代表是子串，所以 j - 1 说明最后一个 t[j] 一定不去匹配。
2. 不用 `s[i]` 来进行匹配，而是用 s[0 ... i - 1] 的子序列来进行匹配 t[j] , 所以为 ： `f[i - 1][j]` 
以上两者相加就是 s[i] == t[j] 的转移函数：`f[i - 1][j] + f[i - 1][j - 1]`

如果 s[i] 不能匹配 t[j] ，则一定不能用 s[i] 来匹配这个字符串，一定是用之前的子序列的结尾来匹配当前的 t[j] ：
`f[i][j] = f[i - 1][j]`

`初始化`：当 t 为空时，即空集一`定会匹配`，且匹配数量为 1 : f[...][0] = 1

`边界`：如果 t.size() > s.size(), 则 s 的子序列一定匹配不了 t, 因为长度不够。

(我们在代码实现时，为了好初始化定义，多增加了一个维度，所以判断的是 s[i - 1] 与 t[j - 1], 而不是 s[i] 与 t[j])

代码如下：
```c++
const int M = 1e9 + 7;
int numDistinct(string s, string t) {
    int m = s.size(), n = t.size();
    if (n > m) return 0;    // `边界`：如果 t.size() > s.size(), 则 s 的子序列一定匹配不了 t, 因为长度不够。
    vector<vector<int> > f(m + 1, vector<int>(n + 1, 0));
    for (int i = 0; i <= m; i ++ ) f[i][0] = 1; // 当 t 为空时，即空集一`定会匹配`，且匹配数量为 1

    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= i && j <= n; j ++ )   // 这里可以用边界条件剪枝，即 t[0...j] 的长度要 <= s[0...i] 否则长度太大，一定不会匹配
            f[i][j] = (s[i - 1] == t[j - 1] ? f[i - 1][j] + f[i - 1][j - 1] : f[i][j] = f[i - 1][j]) % M;
    
    return f[m][n];
}
```

## acwing：272. 最长公共上升子序列
https://www.acwing.com/problem/content/274/

这题想要写出同样需要修改状态集合的定义：定义：(i, j) 为所有 a[1 ~ i] 和 b[1 ~ j] 中以 b[j] 结尾的公共上升子序列的集合。f(i, j) 代表这样的所有集合中的最优解元素的最长长度。

状态如何转移？


## 889. 根据前序和后序遍历构造二叉树
体会此题与 `106. 从中序与后序遍历序列构造二叉树` 和 `105. 从前序与中序遍历序列构造二叉树` 的代码实现的区别：

1. 第一个不同的地方在于，我们是`令 arr1[l1 + 1] 为左子树的根` 因为要防止 l1 + 1 不在区间 [l1, r1] 内，所以要多一个判断是否此时区间长度为 1，即 l1 == r1。
2. 第二个不同点在于，这个 `split 点` 不是划分左右两子树的点，而是后序序列中左子树的右端点。
```c++
unordered_map<int, int> mp;
int n;
TreeNode* DFS(vector<int> &arr1, vector<int> &arr2, int l1, int r1, int l2, int r2){
    if (l1 > r1) return NULL;
    TreeNode *root = new TreeNode(arr1[l1]);
    if (l1 == r1) return root;  // 第一个不同的地方在于这里，因为要防止 l1 + 1 不在区间 [l1, r1] 内
    int split = mp[arr1[l1 + 1]];       // 这也是最大的不同点，就是默认左子树存在，那么左子树的根就是 arr1[l1 + 1] .
    int left_size = split - l2 + 1;     // 这里是这个 `split 点` 不是划分左右两子树的点，而是后序序列中左子树区间的右端点
    root->left = DFS(arr1, arr2, l1 + 1, l1 + left_size, l2, split);
    root->right = DFS(arr1, arr2, l1 + left_size + 1, r1, split + 1, r2 - 1);
    return root;
}
TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
    int n = postorder.size();
    for (int i = 0; i < n; i ++ ) mp[postorder[i]] = i;
    return DFS(preorder, postorder, 0, n - 1, 0, n - 1);
}
```
问：为什么说其实这样的构造方法只是构造了其中可能存在的一个二叉树，而不是唯一的结果呢？

答：因为答案一上来就求左子树根结点的值 `arr1[l1 + 1]`，和左子树结点数 `left_size`，但是`没有考虑是否存在左子树`。

举例：pre: {1,2,3}; post: {3,2,1}，那么可能的二叉树如下：
```
        1               1
    2           与          2
3                               3
```
可以看到，如果用我们上面的代码构造出的一定是左边这个样子的二叉树，因为代码默认了左子树存在。

## 956. 最高的广告牌
这应该属于`另类背包问题`：状态定义：f(i, j) 为前 i 件物品(即钢筋)能组成的 `两个钢支架的高度差值为 j 时` 的两个高度之和。

那么最终的答案就是 `f(n, 0) \ 2` 即当两个刚支架的差值为 0 时，它们的`高度之和的一半` 。(或者你定义 f(i, j) 为两个钢支架的高度差值为 j 时的较长的那个刚支架的长度也可以，那么答案就是 `f(n, 0)`)

关键：这题的关键在于状态的更新，即状态的更新是非常难去想象的，他是像背包问题一样，是多阶段转移问题，但是他的`第二个维度的变化`，以及`初始化赋值`和`边界处理`就非常的巧妙。需要去积累，我们先去考虑第二个维度是如何变化的：
1. 对于当前这个物品，我们有三种选择，第一种选择是丢弃，第二种选择是将他加入较长的那个钢架上，第三种选择是将他加入较短的那个刚架上，那么这三种情况转移方程分别为：`f[i][j] = f[i - 1][j], f[i - 1][j + nums[i]] + nums[i], f[i - 1][abs(j - nums[i])] + nunms[i]` 可以看到，第一种情况下，其差值 j 是不变的，第二种情况，差值增加到 j + nums[i], 第三种情况下，差值减少到 abs(j - nums[i]) 。而后两种情况下需要将这个物品的长度加入到总长度上。

2. 初始化赋值：可能你会误以为，初始化赋值就是：f[i][j] = f[i - 1][j], 这样想是对的，即在数学逻辑上，这样想是没毛病的，但是如果在代码中，只是简单的写上这一句，那么将会导致代码没有正确的实现这个数学逻辑，究其原因就在于，当在处于 j 时，会对别的 `j_others = dif1 or dif2` 差值进行更新。那么当到了你处于 j_others 时，再初始化 `f[i][j_others] = f[i - 1][j_others]` 就会导致原来的更新直接失效。但是如果我们将空间维度进行压缩，即让第一维度进行隐含；或者`先`进行一次`全面的初始化`，即在遍历 j 维度之前，让 f[i] = f[i - 1] (这是对一整个向量进行先行的拷贝) ，然后`不再`进行初始化；或者用 f[i][j] = max(f[i - 1][j], f[i][j]) 来代替原来的赋值操作，使其他的 j_others 对当前 j 的更新有效。这三种方法都是可行的。第二种方法实际上就是弱鸡版的第一种方法，还不如直接使用第一种方法。

3. 边界处理：可以知道，这个`差值最大的情况`为：所有的物品都叠加在一个钢支架上，而另一个钢支架上没有任何物品，那么此时差值最大，为所有物品长度之和 sum。那么`第二个维度大小就是 sum(nums[])`。然后我们考虑一下差值 dif：我们不能让 dif 超过 n，也就是说，当 dif > n 时，我们不应该对其进行更新，但是在遍历第二个维度 j 时，总会遍历到 j == n，那么 j == n 时，dif1 = j + nums[i] 此时绝对会超出范围，该如何避免呢？在遍历到 j 时，我们只去更新那些当前差值 j 情况下的长度总和 f(i, j) >= j 的 f(i, j) ，因为你的差值为 j，那么你的总长度一定要 >= j，而你的总长度若 == j, 那么恰好就处于`刚才所说的插值最大`边界情况，此时一定不会越界。而如果你的总长度 > j，那么你的 dif1 一定不会超过边界情况，因为你已经让之前的物品的长度在两个钢架上都有添加，所以新的 dif 一定不会超过 sum(nums[]) ，所以也不会越界。所以关键的判别越界代码就是：`t[j] >= j` 则进行更新


代码如下：`压缩第一个维度 i 后的代码`
```c++
int tallestBillboard(vector<int>& nums) {
    int m = nums.size();
    int n = accumulate(nums.begin(), nums.end(), 0);
    vector<int> f(n + 1);
    vector<int> t(n + 1);       // 进行维度压缩使用的辅助数组
    for (int i = 1; i <= m; i ++ ) {
        t = f;  // 进行维度压缩的技巧代码
        for (int j = 0; j <= n; j ++ ) {
            if (t[j] < j) continue;     // 一定要保证长度总和 >= j, 同时这样也保证了不会越界
            int dif1 = j + nums[i - 1];
            f[dif1] = max(f[dif1], t[j] + nums[i - 1]);     // 状态转移方程的代码实现，需要用 t 与 f 来更新 f，这里是将物品放在长的钢架上
            int dif2 = abs(j - nums[i - 1]);
            f[dif2] = max(f[dif2], t[j] + nums[i - 1]);     // 将物品放在短的钢架上
        }
    }
    return f[0] / 2;
}
```

## 879. 盈利计划
这题是一个多维度背包问题，还是直接去通过代码来理解。需要增加状态的维度来考虑

定义： f(i, j, k) 为考虑前 i 件物品，使用人数不超过 j，所得利润至少为 k 的方案数。


## 978. 最长湍流子数组
和 `LeetCode 376. 摆动序列` 很像
这题和 acwing 的 zigzag 子序列很相似，只不过将子序列改为了子数组，所以状态定义将`以 i 结尾的子序列`改为`以 i 结尾的子数组`即可。通过经验发现，泽中类型的题目都可以定义为状态机 DP 问题，用状态机 DP 可解决的有：`zigzag 子序列(acwing)`, `926. 将字符串翻转到单调递增`, `股票买卖`，`790. 多米诺和托米诺平铺`等等

直接上状态机 DP 代码：定义 f(i, 0) 为`以 i 为结尾且最后一个元素为上升状的子数组`，f(i, 1) 为`以 i 为结尾且最后一个元素为下降状的子数组`
```c++
int maxTurbulenceSize(vector<int>& arr) {
    int n = arr.size();
    vector<vector<int> > f(n, vector<int>(2, 1));
    int ans = 1;
    for (int i = 1; i < n; i ++ ) {
        if (arr[i - 1] < arr[i])
            f[i][0] = f[i - 1][1] + 1, f[i][1] = 1;
        else if (arr[i - 1] > arr[i])
            f[i][1] = f[i - 1][0] + 1, f[i][0] = 1;
        else // 这里的 else 可以省略，因为在初始化时已经将所有的数组长度改为 1 了
            f[i][0] = f[i][1] = 1;
        ans = max({ans, f[i][0], f[i][1]});
    }
    return ans;
}
```

## 983. 最低票价
可以用记忆化搜索或者递推式来解决，记忆化搜索好写，这里上记忆化搜索的代码，至于递推代码的话，在提交记录里有。

关键：以第 i 天为结尾时的最小费用，或者以第 i 天开始旅行的最小费用来设计状态。
```c++
const int INF = 0x3f3f3f3f;
int S[380] = { 0 };
int f[400] = { 0 };
vector<int> costs;
int DFS(int i) {
    if (i > 365) return 0;
    if (f[i] != -1) return f[i];
    if (!S[i]) return DFS(i + 1);
    int &res = f[i]; res = INF;
    res = min({DFS(i + 1) + costs[0], DFS(i + 7) + costs[1], DFS(i + 30) + costs[2]});
    return res;
}
int mincostTickets(vector<int>& days, vector<int>& _costs) {
    for (auto e : days) S[e] = 1;
    costs = _costs;
    memset(f, -1, sizeof f);
    return DFS(1);
}
```

## 996. 正方形数组的数目
这题可以用`有重复元素的排列 + 剪枝`来做。
```c++
vector<int> v;
vector<int> nums;
vector<int> t;
int ans = 0;
int n;
bool check(int a, int b){
    int sqt = sqrt(a + b);
    if (sqt * sqt == a + b) return 1;
    else return 0;
}
void DFS(int step){
    if (step == n) { ans ++ ; return; }
    
    for (int i = 0; i < n; i ++ ) {
        if (v[i] || i != 0 && nums[i] == nums[i - 1] && !v[i - 1]) continue;
        if (t.size() && !check(t.back(), nums[i])) continue;
        t.push_back(nums[i]);
        v[i] = 1;
        DFS(step + 1);
        t.pop_back();
        v[i] = 0;
    }
}
int numSquarefulPerms(vector<int>& _nums) {
    nums = _nums;
    n = nums.size();
    sort(nums.begin(), nums.end());
    v = vector<int>(n, 0);
    DFS(0);
    return ans;
}
```


## 1000. 合并石头的最低成本

状态定义：DFS(i, j, m) 代表：

从 n 堆变成 1 堆，需要减少 n - 1 堆。而`每次合并都会减少 k - 1 堆`，所以 n - 1 必须是 k - 1 的倍数。即 n - 1 == x * (k - 1) , 其中 x 为整数


`两个关键性问答`：

问 1 :为什么只考虑分出 1 堆和 m - 1 堆，而不考虑分出 x 堆和 p - x 堆?

答:无需计算，因为 m - 1 堆继续递归又可以分出 1 堆和 m - 2 堆，和之前分出的 1 堆组合，就已经能表达出 -> `分出 2 堆和 m - 2 堆`的情况了。其他同理。所以只需要考虑分出 1 堆和 m - 1 堆。

问 2 ：为什么在 DFS 内部中，在`考虑分出` 1 堆和 m - 1 堆 `时`，`无需判断`在 i ~ split 与 split + 1 ~ j 这两部分是否能够分别合法地(即至少一次合并 k 堆)组成 `1` 堆 与 `m - 1` 堆 呢？即无需考虑 `len1 = split - i + 1` 与 `len2 = j - split` 能否合法地合成对应的 1 堆 与 `m - 1` 堆呢？

答：因为我们在枚举时`保证了 split 取到某些特定的值`，使得分出来的这两个部分`是一定合法且不重不漏`的。这是考虑了离散数学中的 k 叉数的思想，证明略，但是可以证明，这样枚举出来是一定合法且不会遗漏任何一种情况的。

## 1012. 至少有 1 位重复的数字
此题与 `2376. 统计特殊整数` 相同，我是利用了 repeat 作为 special 参数，来描述 t 这个数字是否存在重复，然后让参数 isNum 和 isLimit 全部不做记忆化，让 mask 和 repeat 做了记忆化，这样时间复杂度较高，直接利用 `2376. 统计特殊整数` 的算法，即利用数学的求反思想来考虑，即统计全部不重复的数字数量 n1，那么重复的数字数量应该为 `n2 = 总数量 - n1 = n - n1`;



## 1014. 最佳观光组合
这题的做法非常类似于 `53. 最大子数组和` 的`前缀和基本解法`，即维护一个最小的 `min_pre = -A[i] - i` 或者是维护最大的 `max_pre = A[i] + i` ，然后同时遍历整个数组，进行 `ans = (A[j] - j) - min_pre` 或者 `ans = (A[j] - j) + max_pre`

即拆解原式后分别得到两个一元函数，然后就可以通过一次遍历解决问题。代码风格与  `53. 最大子数组和` 的`前缀和基本解法` 极其类似，我就是通过这个思路想到的。

代码：
```c++
int maxScoreSightseeingPair(vector<int>& nums) {
    int ans = 0, n = nums.size();
    int min_pre = -nums[0];
    for (int i = 1; i < n; i ++ )
        ans = max(ans, nums[i] - i - min_pre), min_pre = min(min_pre, -nums[i] - i);
    return ans;
}
```

## 1024. 视频拼接
解法一：动态规划做法

这题就是弱鸡改版的 `139. 单词拆分`，其状态定义和转移方式很类似：定义 `f(i)` 为视频时间从 0 ~ i 拼接成功所需最少的 `片段clips`。

动态规划转移方程：`f[i] = min(f[i], f[clips[j][0]] + 1);` 解释：即遍历所有的片段 clips 且要`符合 i 处于区间 (clips[j][0], clips[j][1]] 左开右闭`，去更新 f(i) ，取片段的开头 `clips[j][0]` 作为上一个状态的转移位置，然后只需要`增加一个片段，即这个片段`，所以是 `f[clips[j][0]] + 1`。然后取最小值即可。

注：判断条件中也可以是 `左闭右闭` 区间，没有关系，因为如果是左闭区间的话，实际上是：f[i] = min(f[i], f[i] + 1) 即实际上不会被更新。

初始化及其边界：由于我们转移方程为取最小值，所以我们让 f(1 ~ time) 初始化为 INF，而对于 f(0) 直接赋值为 0，即默认他是不需要拼接即可转移的。然后即可正确转移，代码如下：

```c++
const int INF = 0x3f3f3f3f;
int videoStitching(vector<vector<int>>& clips, int time) {
    vector<int> f(time + 1, INF);
    int n = clips.size();
    f[0] = 0;
    for (int i = 1; i <= time; i ++ ) 
        for (int j = 0; j < n; j ++ ) 
            if(i <= clips[j][1] && i > clips[j][0]) // 符合 i 处于区间 (clips[j][0], clips[j][1]] 左开右闭
                f[i] = min(f[i], f[clips[j][0]] + 1);
        
    
    if (f[time] != INF) return f[time];
    else return -1;
}
```

解法二：模拟 -> 跳跃游戏解法。
我们令 maxLen 为当前这个区间能够跳跃的最远距离，然后令 curR 为当前选中的区间的右端点，而当前区间是隐含的，即当前区间的左端点是`在上一个的区间遍历中的某一个点，这个点更新了最大的 maxLen ，但是不用记录这个这个点`，而右端点则是当前的 curR。

算法：
一、预处理出每个点的最大跳跃点：
1. 遍历每一个 clips 中的元素 e，并只更新左区间端点 < time 的点，因为`不必要去考虑`端点 >= time 的片段，这些`片段``不可能`去`用于拼凑`从 [0, time] 区间的视频` 

二、遍历所有的下标 i：
1. 更新`在当前区间内`能够到达的最远距离 maxLen = max(maxLen, nums[i])
2. 如果在`更新后`其最原距离就是当前的位置 i，说明他没有移动，说明他就动不了了，返回 -1。这也`保证了`当前位置至少要往前移动一个位置，否则直接返回 -1.
3. 如果`当前位置 == 当前区间的右端点`则将 curR 更新为在`这段区间`中能更新为的最远位置 maxLen，即相当于`选中了下一个当前区间`。由于选中了下一个当前区间， 所以需要 ans ++ ，即每次选了下一个区间则将`答案 + 1`。

代码：
```c++
int videoStitching(vector<vector<int>>& clips, int time) {
    vector<int> nums(time);
    for (auto &e : clips)
        if (e[0] < time)
            nums[e[0]] = max(nums[e[0]], e[1]);
    
    int maxLen = 0, curR = 0, ans = 0;
    for (int i = 0; i < time; i ++ ) {
        maxLen = max(maxLen, nums[i]);
        if (i == maxLen) return -1;
        if (i == curR) ans ++ , curR = maxLen;
    }
    return ans;
}
``` 

## 1027. 最长等差数列
这题的`状态定义`要借鉴该题：`等差数列划分 II` 与 `956. 最高的广告牌`，即`将公差 d 作为状态中的一个维度`来定义状态，而`状态转移`方程借鉴了 `最长递增子序列`。

我们定义 f[i][d] 表示以 nums[i] 结尾且公差为 d 的等差数列的最大长度。初始化所有 f[i][d] = 1 ，即每个元素自身都是一个长度为 1 的等差数列。

那么状态转移方程为：`f[i][d] = max(f[i][d], f[j][d] + 1)`  其中 `j in [0, i - 1]` 即在 nums[i] 之前的所有 nums[j] 都去考虑对应的 d，类似于最长递增子序列。

代码实现的两个技巧：
1. 用 `数组 + 偏移量` 代替 `哈希表` : 我们设置一个偏置值 ofst = 500，因为 d 可能为负数，而差值的绝对值最大不超过 500 ，所以我们直接用 d + ofst 来作为第二个维度，省去了使用哈希表带来的极大的空间复杂度和时间复杂度。
2. 初始化所有 `f[i][d] = 1` 改为 -> `f[i][d] = 1`, 那么答案改为原答案加一即可：` ans + 1 `

关键点：


代码如下：
```c++
int longestArithSeqLength(vector<int>& nums) {
        int ofst = 510;
        int n = nums.size();
        vector<vector<int> > f(n, vector<int>(1100, 1));
        int ans = 1;
        for (int i = 1; i < n; i ++ ) {
            for (int j = i - 1; j >= 0; j -- ) {
                int dif = nums[i] - nums[j] + ofst;
                f[i][dif] = max(f[i][dif], f[j][dif] + 1);
                ans = max(ans, f[i][dif]);
            }
        }
        return ans;
    }
```

## 1031. 两个非重叠子数组的最大和
解法：前缀和 + 滑动窗口(同向双指针 + 长度`固定`型)

先解决前置问题：`209. 长度最小的子数组` 此题的解法为: 前缀和 + 滑动窗口(同向双指针 + 长度`非固定`型)。由于长度是非固定型，所以`左端点 l 需要在 r 固定时，进行移动`。

同向双指针的要求就是：l 与 r 的方向同向增加，由于这是前缀和，所以`为了符合数学定义`，我们所维护的区间`不是 [l, r]`，`而是 (l, r], 即 [l + 1, r]`。因为前缀和的 `s[r] - s[l]` 时，是`没有包含 nums[l] 在区间内`的，而是`从 l + 1 ~ r 的 nums 的和`

直接上代码，其中`有几个小细节`注释在代码中：
```c++
const int INF = 0x3f3f3f3f;
int minSubArrayLen(int target, vector<int>& nums) {
    int l = 0, r = 1, n = nums.size(), ans = INF;   // 注意这里必须初始化 ans = INF，防止最大子数组和 < target
    int s[n + 1]; s[0] = 0;
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];
    for ( ;r <= n; r ++ ) {
        while (l < r && s[r] - s[l + 1] >= target) 
            l ++ ;
        if (s[r] - s[l] >= target) // 这是为了保证当窗口在 `较小` 时，如果不满足窗口内子数组和 >= target 则不能更新答案
            ans = min(ans, r - l);
    }
    return ans == INF ? 0 : ans;    // 由于可能没有大于等于 target 的子数组
}
```

解决了前置问题之后，我们来解决这个问题：前缀和 + 滑动窗口(同向双指针 + 长度`固定`型)

由于这是`长度固定型`滑动窗口，所以我们使用的数据结构`不再是`两个`左右指针`，而是 `窗口长度 len` + `右端点指针 r`。那么此时维护的区间段为：`[r - len + 1, r]` 

两个关键问题：

问 1：如何解决这两个不同大小的滑动窗口的相对位置问题，即哪个在哪个左侧，哪个在右侧？

答：进行`分两个类讨论`，计算两次，一次是长度为 a 的在左侧，一次是长度为 b 的在左侧。答案取分类讨论中的最大值

问 2：如何解决在 b 窗口更新的同时，保证 a 窗口也会实时更新？即如何保证两个窗口在遍历数组时同时进行更新？

答：我们在 b 窗口向右滑动的同时，可得出 b 窗口的左端点，然后`利用 b 窗口的左端点 b_left` 来作为 a 的窗口右端点，然后通过 a 的当前窗口大小是否大于 a 的旧窗口的最大值来`决定是否更新 a 的旧窗口`(这个旧窗口是从 0 ~ b_left 中子数组和最大的旧窗口)。实际上`可以理解为动态规划的空间压缩`，即用一个 maxS_a 来记录从 0 ~ b_left 的最大长度为 a 的值 -> f(b_left)。

```c++
int s[1010], n;
int f(int a, int b) {
    int maxS_a = 0, ans = 0;
    for (int i = a + b; i <= n; i ++ ) {
        maxS_a = max(maxS_a, s[i - b] - s[i - a - b]);
        ans = max(ans, maxS_a + s[i] - s[i - b]);
    }
    return ans;
}
int maxSumTwoNoOverlap(vector<int>& nums, int a, int b) {
    n = nums.size();
    s[0] = 0;
    for (int i = 1; i <= n; i ++ ) s[i] = s[i - 1] + nums[i - 1];
    return max(f(a, b), f(b, a));
}
```


## 1035. 不相交的线
此题直接转化为最长公共子数组，非常巧妙。分析如下：

k 条互不相交的直线分别连接了数组 nums1 和 nums2 的 `k 对相等的元素`，而且这 k 对相等的元素在两个数组中的`相对顺序是一致`的，因此，`这 k 对相等的元素组成的序列即等价于数组 nums1 和 nums2的公共子序列`。要计算可以绘制的最大连线数，即为计算数组 nums1 和 nums2 的最长公共子序列的长度。

直接上代码：
```c++
int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size();
    int f[m + 1][n + 1]; memset(f, 0, sizeof f);
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ ) 
            f[i][j] = nums1[i - 1] == nums2[j - 1] ? f[i - 1][j - 1] + 1 : max(f[i - 1][j], f[i][j - 1]);
        
    return f[m][n];
}
```

## 1039. 多边形三角剖分的最低得分
此题是 `数学几何` + `区间DP` 的结合，其中数学几何是最为重要的部分。

数学几何：在多边形的两个`顶点` i 与 j 之间(`不包括` i 点与 j 点) 我们选取一个顶点 k，那么我们就在 <i, k> 与 <k, j> 这两对顶点之间连上线：即可分割为`一个三角形(顶点 <i k j> 组成的)和两个多边形`这`三部分`，然后我们就可以递归地进行遍历这两个多边形，而分割出的这个三角形的得分我们可以直接算出来，总的来说，如果用 DFS 记忆化搜索会比递推的方式好做，因为不用考虑如何去进行代码实现递推，而 DFS 的代码好写的很。

那么我们就可以写出转移方程：在 l 与 r 之间的多边形内进行选出 k 来进行区间 DP：
`res = min(res, DFS(l, k) + DFS(k, r) + nums[l] * nums[k] * nums[r])` 

边界处理：如果两个顶点紧挨着就构不成三角形，这种情况下：`l + 1 == r`
```c++
const int INF = 0x3f3f3f3f;
int memo[55][55];
int n;
vector<int> nums;
int DFS(int l, int r){
    if (l + 1 == r) return 0;
    if (memo[l][r] != -1) return memo[l][r];

    int &res = memo[l][r]; res = INF;
    for (int k = l + 1; k < r; k ++ )
        res = min(res, DFS(l, k) + DFS(k, r) + nums[l] * nums[k] * nums[r]);
    
    return res;
}
int minScoreTriangulation(vector<int>& values) {
    nums = values;
    n = nums.size();
    memset(memo, -1, sizeof memo);
    return DFS(0, n - 1);
}
```

## 1043. 分隔数组以得到最大和
同样是记忆化搜索比递推写法好做。

此题的转移方式类似于最长递增序列，但是处理方式更加巧妙。转移方程更加难想，和 `1105. 填充书架` 一样难想，即 `合法性构造` 最长递增子序列问题。我们需要`构造一个段长度的合法区间(合法即长度应该小于等于 k )` 的符合条件的位置处进行转移。填充书架也是构造一个合法的段长进行转移

定义：DFS(i) 表示`将数组的前 i 个元素分隔成若干个子数组，并通过操作使得整个数组最终的最大元素和`。在边界时： DFS(-1) = 0，答案为 DFS(n - 1)。`注`：为什么`不能`让边界 `DFS(0) = arr[0]` ？ 这是因为我们可以`通过题目中允许的操作``让 arr[0] 变为更大的数`，所以不能简单的返回 arr[0].

`特别注意`的是：这个转移的方程是在一个区间`左端点之前`的那个端点进行转移，`而不是`这个区间的左端点。所以这点非常重要，这种方式很像这个做法：
用 `单调队列滑动窗口 + 前缀和` 来处理长度最大为 k 的子数组和，这个滑动窗口就是当前点的前面的一个窗口，而不是当前的那个窗口。

转移方程：res = max(res, DFS(j - 1) + mx * (i - j + 1)) 我们维护一个窗口，该窗口区间为 `[j, i]` 右端点固定为 i，那么这个窗口`最大为 k`，所以区间最大时为 `[i - k + 1， i]` 而左端点应该 >= 0，所以 j 的遍历就是在区间：`[max(0, i - k + 1), i]` 然后我们应该在 j 的`前面`那个`端点 j - 1 进行转移` ：`res = max(res, DFS(j - 1) + mx * (i - j + 1))`。其中 mx 就是转移方程很难想的地方：mx 就是 j 在区间内从右到左遍历时经过的最大值，因为我们的操作就是将区间内的所有点都转化为 mx。

代码如下：
```c++
int memo[510];
int n, k;
vector<int> arr;
int DFS(int i) {
    if (i == -1) return 0;
    if (memo[i] != -1) return memo[i];
    int &res = memo[i], mx = 0; res = 0;
    for (int j = i; j >= max(0, i - k + 1); j -- )
        mx = max(mx, arr[j]),
        res = max(res, DFS(j - 1) + (i - j + 1) * mx);
    return res;
}
int maxSumAfterPartitioning(vector<int>& _arr, int _k) {
    arr = _arr, k = _k, n = arr.size(); memset(memo, -1, sizeof memo); 
    return DFS(n - 1);
}
```

## 1049. 最后一块石头的重量 II   数学逻辑分析 + 动态规划
此题就是 01 背包问题，由于需要`很难的数学逻辑转换方法`所以这题显得比较难，定义：f[i][j] 为前 i 件石头能组成的最大的重量和。

令 `sum 为所有石头的总和`，那么答案就是：  `abs(f[n][sum / 2] - (sum - f[n][sum / 2])) = sum - 2 * f[n][sum / 2]`

数学转换的两个关键思考步骤：

转换 1 ：题目等价于求：我们将所有石头分为两组，然后使得这两组的石头的重量的差值尽可能的小。那么最终的答案就是这两组的石头重量和的差值。问：为什么能这样转换呢？

答：
1. 我们由题目可以知道(`关键`)：经过 n 次粉碎后，`最终最多只会剩下 1 个石头`，并且我们需要`让最后一块石头的质量最小`
2. 我们继续分析可以发现：我们可以将这一堆石头分成两堆（ heap1 和 heap2 ）
3. 我们不妨设 heap1总质量 >= heap2总质量，而最后的结果就是heap1 - heap2，我们只需要保证heap1 - heap2最小即可

问：最后的结果为什么是 `sum - 2 * f[n][sum / 2]` 答：由于 `sum / 2 = floor(sum / 2.0)` 是较小的那一堆，那么我们设较小的那一堆为 x ( x 就是    `f[n][sum / 2]` )，另一堆就是 sum - x，而 2 * x < sum，所以两个石堆之差为 `sum - 2 * x = sum - 2 * f[n][sum / 2]`

代码：
```c++
int lastStoneWeightII(vector<int>& nums) {
    int n = nums.size(), sum = 0;
    for (int i = 0; i < n; i ++ ) sum += nums[i];
    int tar = sum / 2, f[tar + 1]; memset(f, 0, sizeof f);
    for (int i = 0 ; i < n; i ++ )
        for (int j = tar; j >= nums[i]; j -- )
            f[j] = max(f[j], f[j - nums[i]] + nums[i]);
    return sum - (f[tar] << 1);
}
```

## 1105. 填充书架
这题两个值得学习的点：
1. 约束条件同转移方程写出的关联
2. `语法模板格式`：在 for 中，有 break 时，`跳出条件执行时`，应该放在代码的哪个位置才能不会使下标越界。

这题的`关键`就是两个约数条件和`如何求转移方程`，只有利用这两个约束条件才能写出动态规划方程：
1. 必须按规定的书籍顺序来填充书架
2. 同一层的书籍宽度不得超过暑假宽度

很多时候不会动态规划的原因就是不会定义边界和转移方程，我们先来定义状态和转移方程：
1. 定义 DFS(i) 为前 i 本书能凑出的书架最小层数。
2. 那么我们应该这样转移：从第 i 本书开始，从后往前枚举书本 j，将 `[j, i] 区间内` 的所有书都竖着叠起来，记录它们的总厚度 thick，然后在不超过书架厚度 width 的情况下，我们将它们视为放在书架的同一层，那么转移方程为：`res = min(res, DFS(j - 1) + max_h)`  其中 max_h 为 [j, i] 区间内最最高的书的高度。正确性证明：由于我们是`按顺序`进行填充书架，所以正确的摆放方法一定是从小到大，所以我们这样枚举 j 就保证了让 [j, i] 区间内放同一层时，让`前一层的高度` DFS(j - 1) 最小：因为 [j, i] 是同一层，所以 j - 1 就是上一层。

边界条件：`DFS(-1) = 0` 因为下标为 -1 时，我们没有书可以填充，所以可以让书架为 0.

代码如下：
```c++
int memo[1010];
int n, width;
vector<vector<int>> books;
const int INF = 0x3f3f3f3f;
int DFS(int i){
    if (i == -1) return 0;
    if (memo[i] != -1) return memo[i];
    int &res = memo[i], thick = 0, max_h = 0; res = INF;
    for (int j = i; j >= 0; j -- ) {        // 如何在 for 的第二个语句中写 thick 的判断条件而不导致下标越界？ 答：将 j >= 0 放在前面，将判断放在后面，这样才能保证不越界。所以以后遇到这种东西我们一定要记住，必须将下标判断放在条件判断的前面！！例：
    // for (int j = i; j >= 0 && (thick += books[j][0]) <= width; j -- )
        thick += books[j][0], max_h = max(max_h, books[j][1]);
        if (thick > width) break;
        res = min(res, DFS(j - 1) + max_h);     
    }
    return res;
}
int minHeightShelves(vector<vector<int>>& _books, int shelfWidth) {
    books = _books, n = books.size(); width = shelfWidth;
    memset(memo, -1, sizeof memo);
    return DFS(n - 1);
}
```

## 1130. 叶值的最小代价生成树
此题和 `1105. 填充书架` 一样，有一个很重要的条件需要注意：数组 arr 中的值与`树的中序遍历`中每个`叶节点`的值一一对应。也就是说，arr 中每个元素`都是叶节点`。

那么我们可以定义状态：DSF(i, j) 为区间 [i, j] 内所有`非叶节点(即分支节点)值`和的最小值(即二叉树的形状为这个最小值时特定的取到)。那么：
1. i == j 时，i 和 j 就是同一个叶节点，于是它们 `没有分支节点` 所以我们直接返回 0
2. i != j 时，我们划分为两个区间：`[i, k] 与 [k + 1, j]` 那么 arr 的这两部分叶节点就被划分为了`左子树的叶节点`和`右子树的叶节点`。那么转移方程为： `ans = min(ans, dfs(i, k) + dfs(k + 1, j) + g[i][k] * g[k + 1][j])` 其中 g[i][k] 为左区间内的最大值，g[k + 1][j] 为右区间内的最大值。

所以由上述分析可知，我们应该去先预处理出 `g[l][r] 即区间 [l, r] 内的最大值`。

```c++
int g[50][50];
int n, memo[50][50];
const int INF = 0x3f3f3f3f;
int DFS(int l, int r) {
    if (l == r) return 0;
    if (memo[l][r] != -1) return memo[l][r];
    
    int &res = memo[l][r]; res = INF; 
    for (int k = l; k < r; k ++ ) 
        res = min(res, DFS(l, k) + DFS(k + 1, r) + g[l][k] * g[k + 1][r]);
    
    return res;
}
int mctFromLeafValues(vector<int>& arr) {
    n = arr.size();
    memset(g, 0, sizeof g); memset(memo, -1, sizeof memo);
    for (int i = n - 1; i >= 0; i -- ) {
        g[i][i] = arr[i];
        for (int j = i + 1; j < n; j ++ )
            g[i][j] = max(g[i][j - 1], arr[j]);
    }
    return DFS(0, n - 1);
}

```
## 1139. 最大的以 1 为边界的正方形
此题亦是几何形状类DP问题，类似问题：`221. 最大正方形`，只不过此题求的`是允许空心`的正方形，而 `221. 最大正方形` 求的是实心的正方形。

我知道要用前缀和来解决该问题，可关键是如何用代码进行递推的去进行动态规划转移，并如何来进行代码模拟。

方法 1 ：利用以坐标 (x, y) 为结尾，连续的 1 的个数来进行模拟：

定义：记 f(i, j) 为以 (i, j) 为结尾，向左能最大延申的连续 1 的个数(`称之为臂展`) ，记 g(i, j) 为以 (i, j) 为结尾，向上的最大臂展。

那么我们就写出动态规划方程，以 (i, j) 为右下角的正方形，最大能是多少?

我们`从大到小`枚举正方形的边长 side ：然后去考虑坐标 (i, j - side + 1) 向左的最大臂展，和考虑坐标 (i - side + 1, j) 的最大向上的臂展。判断是否能组成空心正方形即可。注意臂展 len 一定是 `len >= side` 即可成立，而不是 `==`

代码：
```c++
int largest1BorderedSquare(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    int f[m + 1][n + 1], g[m + 1][n + 1]; 
    memset(f, 0, sizeof f); memset(g, 0, sizeof g);
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ )
            f[i][j] = mat[i - 1][j - 1] == 1 ? f[i][j - 1] + 1 : 0,
            g[i][j] = mat[i - 1][j - 1] == 1 ? g[i - 1][j] + 1 : 0;
    int ans = 0;
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ )
            for (int side = min(f[i][j], g[i][j]); side > 0; side -- ) 
                if (f[i - side + 1][j] >= side && g[i][j - side + 1] >= side) {
                    ans = max(ans, side);
                    break;
                }
    return ans * ans;
}
```

也可以用前缀和来写，其逻辑实际上是一模一样，但是枚举方式却是先枚举 side 再枚举正方形 `左上角下标` 是否满足一个正方形即可。
```c++
int largest1BorderedSquare(vector<vector<int>> &grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<int>> s1(m, vector<int>(n + 1)), s2(n, vector<int>(m + 1));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            s1[i][j + 1] = s1[i][j] + grid[i][j]; // 每行的前缀和
            s2[j][i + 1] = s2[j][i] + grid[i][j]; // 每列的前缀和
        }
    for (int d = min(m, n); d; --d) // 从大到小枚举正方形边长 d
        for (int i = 0; i <= m - d; ++i)
            for (int j = 0; j <= n - d; ++j) // 枚举正方形左上角坐标 (i,j) 如果四测的边长都等于 side 则找到了最大的 side
                if (s1[i][j + d] - s1[i][j] == d && // 上侧边长
                    s2[j][i + d] - s2[j][i] == d && // 左侧边长 
                    s1[i + d - 1][j + d] - s1[i + d - 1][j] == d && // 下侧边长
                    s2[j + d - 1][i + d] - s2[j + d - 1][i] == d)   // 右侧边长
                    return d * d;
    return 0;
}
```

## 1155. 掷骰子等于目标和的方法数
此题就是一个分组背包问题(分组背包问题详解见算法笔记)，而其最难的地方在于初始化最初状态：

我们令 f(i, j) 代表前 i 个骰子能凑出的点数为 j 的全部方案数，那么状态转移方程很好写，利用分组背包的模板即可。值得注意的是：这里必须让 k = 1 ，即第 i 个骰子`一定要选取`，`不可以不选`！！而正是因为这个原因，导致初始化会与分组背包的初始化不同，因为`背包是可选可不选`，而这个`掷骰子是一定要选`。所以在初始时，我们让 `f(0, 0) = 1`, 这是因为为了`你第一个骰子无论掷了多少点 x，能从 (0, 0) 处转移到 (1, x) 此时方案数一定是 1 个`。但是 `f(1 ~ n - 1, 0) = 0` 这是因为如果你的`掷的骰子不为 0 个`，那么你的`点数不可能为 0`，因为每个骰子的`点数都大于 1`!

总结：所以这还是通过题目的骰子点数条件等等限制出的`初始化`和`转移方程`，这种转换思路是有点难度的。

代码如下：
```c++
const int MOD = 1e9 + 7;
int numRollsToTarget(int n, int s, int target) {
    int f[n + 1][target + 1];
    memset(f, 0, sizeof f);
    f[0][0] = 1;
    for (int i = 1; i <= n ; i ++ )
        for (int j = 1; j <= target; j ++ )
            for (int k = 1; k <= s; k ++ )  // 一定要选，所以 k 必须从 1 开始遍历
                if (j >= k)    // 值得注意的是，可以把这个条件放入 for 的中间语句中，这是因为不像普通 `无序` 的分组背包，这个分组背包的体积是按顺序递增的，所以一旦不满足体积条件，那么后面的物品更加不满足体积条件
                    f[i][j] = (f[i][j] + f[i - 1][j - k]) % MOD;
    return f[n][target];
}
```

来看一下记忆化搜索版本的代码，直接省去了两重循环：
```c++
const int MOD = 1e9 + 7;
int s, tar;
int memo[33][1010];
int DFS(int i, int j){
    if (i == 0) return i == j ? 1 : 0;
    if (j == 0) return 0;
    if (memo[i][j] != -1) return memo[i][j];

    int &res = memo[i][j]; res = 0; 
    for (int k = 1; k <= s && k <= j; k ++ )    // 可以看到，我们无序对 i 和 j 进行循环，只要对 k 来进行遍历寻找
        res = (res + DFS(i - 1, j - k)) % MOD;
    return res;
}
int numRollsToTarget(int n, int k, int target) {
    s = k, tar = target; memset(memo, -1, sizeof memo);
    return DFS(n, target);
}
```

## 462. 最小操作次数使数组元素相等 II
该题是一个 `数学证明 + 贪心` 问题，其做法是先将 `nums[]` 进行排序，然后我们将每个元素都进行操作，使之变为 `nums[] 的中位数(即 target)` 即可。 如果 nums[] 中的元素数量为偶数，则中间位置的两侧的数任选一个作为 `target` 即可(注：此时中位数在题目条件下一定是小数，double 型)。

证明：
1. 当 nums[] 为奇数时，举例：给出一个已经排好序的测试样例 `[0, 1, 2, 6, 8]` 首先我们考虑首位 0 与 8 两个数，我们让这两个数操作后变为 x(下面`简称为移动到 x`) 那么我们的 x 在 `[0, 8] 区间` 外部时，`易得`移动次数一定大于 x 在区间内部。而 x 在区间内部时，`两者移动次数之和`一定是`固定`的，其值为 `(x - 0) + (8 - x) = 8` 所以无论在 [0, 8] 内部 x 取什么值，其移动次数之和一定是 8所以我们不再关心 0 和 8 这两个数，而去讨论 1 与 6 这两个数：同理，我们应该在 `[1, 6] 区间内部` 去找一个数。然后依次类推，2 就是我们要找的目标值 target = x。
2. 当 nums[] 为偶数时，根据上面的讲述的方法同理可得 `nums[] 数组` 中间位置的两侧的数任选一个作为 `target` 即可

代码：
```c++
int minMoves2(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    int n = nums.size(), tar = nums[n / 2], ans = 0;
    for (int i = 0; i < n; i ++ )
        ans += abs(tar - nums[i]);
    return ans;
}
```
总解：很多这种求最小 `+1 or -1 操作次数`的题目实际上都是贪心来求，而我们一般需要 `利用数学归纳法的逻辑` 从`特殊到一般地` 解决这种问题。

## 2673. 使二叉树所有路径值相等的最小代价
类型：`贪心 + 自底向上树形DP`，这题的思考方法也很秒，也是`从特殊到一般(从叶子节点依次往上一层的分支节点)`的一种思想。

我们应该有一下两个贪心的结论，具体证明不详细分析：
1. 不应修改`从根到某一个叶子`具有最大成本的路径。因为我们可以通过`增加其他根到叶的路径`进行 +1 操作，而最大的再进行 +1 操作则是多余操作，不是最优解。
2. 所以最优的方法是增加所有其他路径，使其成本等于成本最大的路径

那么我们如何实现上面分析的第二条：`增加所有其他路径，使其成本等于成本最大的路径`呢？其`代码实现`非常有意思，是自底向上的树形DP实现模式，我使用的递归方法很难做到当我们在其中一条路径上的某个节点增加元素值的同时，影响到其他所有的经过该节点路径同时增加这一个值。即我用递归算出了所有的路径，但是在对某一节点进行增加操作时，在其他路径该如何去增加，这还是很难写出这个代码的，如果不是按照这样的逻辑写则会出错，因为你增加了某一节点的值会影响到经过该节点的所有路径的和。所以我们要用到树形DP来对这个路径进行累加，即用转移的方法来实现对其中一个分支节点操作后，对其路径都进行加的操作，这个是一个技巧。

1. 首先，我们`并不去计算最大的那条路径和`，我们先让`最后一层的`所有的叶子节点都进行增加到和它兄弟一样大。
2. 然后，我们将叶子节点的上一层节点，即`倒数第二层的节点`的值加上自己的叶子节点的值，注意加一个叶子节点就行，因为经过上一步操作后，叶子节点的值都相同
3. 然后我们将这倒数第二层视作叶子节点，让 `"倒数第二层的"` 所有的叶子节点都进行增加到和它兄弟一样大。
4. 然后我们又去对倒数第三层的叶子节点进行同样的操作，直到根节点。

`注意`：`下标细节`：

对于某个节点 x(编号从 1 开始)，它的左孩子为 x * 2, 右孩子为 x * 2 + 1, 在数组的下标分别为`其编号 - 1`，对于满二叉树来说，其最大的分支节点为： n / 2 (节点从 1 开始编号)。

代码如下：
```c++
int minIncrements(int n, vector<int>& cost) {
    int ans = 0;
    for (int i = n / 2; i > 0; i -- ) {
        ans += abs(cost[i * 2 - 1] - cost[i * 2]);
        cost[i - 1] += max(cost[i * 2 - 1], cost[i * 2]);
    }
    return ans;
}
```
## 2208. 将数组和减半的最少操作次数
此题为：`优先队列贪心` 的典型问题，即用一个优先队列来维护一个当前数组内的所有元素，并且最大值在队首(即堆顶)。然后我们每次都在堆顶进行选取元素，`累计减少的值，知道超过总和的一半`。

```c++
int halveArray(vector<int>& nums) {
    priority_queue<double> q(nums.begin(), nums.end());
    double sum = 0;
    for (int i = 0 ; i < nums.size(); i ++ ) sum += nums[i];
    double tar = sum / 2.0, inc = 0;
    int ans = 0;
    while (inc < tar) {
        auto t = q.top(); q.pop();
        t /= 2; q.push(t);
        inc += t;
        ans ++ ;

    }
    return ans;
}
```
## 2870. 使数组为空的最少操作次数
此题是一个 `哈希表 + 数学模拟分类讨论贪心` 问题。我们用哈希表来存储数组的值相等的元素的个数，其键对为：`<value, cnt>` 含义：元素值为 value 的元素共有 cnt 个。那么对于一个 value 来说，我们分情况讨论：
1. 如果其对于的个数 cnt == 1 则无法操作，返回 -1
2. 如果 cnt 恰好可以整除 3 则可以用 cnt / 3 次操作删除
3. 如果 cnt 除 3 余 1 则它`一定形如：cnt == 1 + 3 + 3 + 3 + ... + 3` 那么我们应该将他先进行 `(cnt / 3) - 1` 次删除第二个 3 开始以后的所有 3 ，然后进行 `2` 次删除 2(即删除前面的 1 + 3)。 共 cnt / 3 + 1 次删除操作
4. 如果 cnt 除 3 余 2 则形如：`cnt == 2 + 3 + 3 + 3 + ... + 3` 那么我们应该进行 `(cnt / 3)` 次删除从第一个 3 开始往后的所有 3 . 然后用一次操作来进行删除`第一个 2`。 共 cnt / 3 + 1 次删除操作.

实际上，上述情况 2，3，4 都可以用一个代码来进行归纳：`(cnt + 2) / 3`

代码如下：
```c++
int minOperations(vector<int>& nums) {
    unordered_map<int, int> mp;
    int n = nums.size();
    for (int i = 0; i < n; i ++ ) mp[nums[i]] ++ ;
    int ans = 0;
    for (auto [value, cnt] : mp) {
        if (cnt == 1) return -1;

        // 实际上以下三种情况可以用一行代码来归纳：ans += (cnt + 2) / 3
        if (cnt % 3 == 0) ans += cnt / 3;   
        else if (cnt % 3 == 1) ans += cnt / 3 + 1;
        else ans += cnt /3 + 1;
    }
    return ans;
}
```

## 2598. 执行操作后的最大 MEX
此题为 `同余数学 + 哈希表 + 枚举贪心` 问题，其存储方法和 `2870. 使数组为空的最少操作次数` 极其类似

我们用哈希表的键值 `<value, cnt>` 代表：数组中所有同余 m 等于 value 的元素数量。

然后进行模拟从小到大来`枚举` `MEX` 可以是什么：
1. MEX 可以为 0 吗？如果 mp[0 % m] > 0 则 0 一定可以生成，所以不能为 0，然后由于`生成 0` 需要`用掉一个同余为 0 的元素`，所以 mp[0] --；同时`枚举下一个 MEX `即 ans ++ 
2. MEX 可以为 1 吗？如果 mp[1 % m] > 1 则 1 一定可以生成，所以 。。。
3. 重复以上枚举，直到 mp[ans % m] == 0 代表不能生成这个 ans 了，说明 MEX 一定是 ans。

代码实现：
1. 技巧一：如何处理初始元素为负数的情况？利用代码：(ele % m + m) % m 这样可以避免直接使用 ele % m 时出现负数的情况
2. 技巧二：用一个 while() 即可实现枚举过程，详见代码
```c++
int findSmallestInteger(vector<int>& nums, int m) {
    unordered_map<int, int> mp;
    int n = nums.size();
    for (int i = 0; i < n; i ++ ) mp[(nums[i] % m + m) % m] ++ ;
    
    int ans = 0;
    while (mp[ans % m] -- )
        ans ++ ;
    return ans;
}
```

## 1162. 地图分析
方法一：对每个节点进行暴力 `单源点 BFS`

该方法时间复杂度为 O(n^4) 因为共有 O(n^2) 个点，且每个点进行一次单源 BFS 需要 O(n^2) 的时间复杂度，所以这是一定会超时的。

也就是用 `单源 BFS` 暴力的 求出每个海洋的最近陆地，

方法二：`多源最短路`

这题积累出的知识以及补缺非常的重要：

概念：啥是超级源点？其实我们可以认为`超级源点`就是`多源点BFS`因为如果有超级源点，那么代表一定有一个点集 S，在集合 S 中任何点到超级远点 src 的距离都相等且为 0，所以可以看作是多源点最短路

知识一：不是说必须要用邻接表或者是邻接矩阵才能用来表示一个图的数据结构，实际上任何一个能够有这两种性质的东西都能称之为图：
1. 性质 1：能够对一个节点进行标识
2. 性质 2：知道一个节点的标识号的情况下，`通过该标识`能够知道该节点的出边与入边，以及该边指向的节点和该边的长度。

所以一个矩阵 mat 同样拥有上述两个性质，可以表示一个图：其对节点的表示为该节点在矩阵中的坐标，其出边为上下左右四个方向。

那么我们将此题转化为 `超级源点 + 最短路问题`：将任何一个陆地到海洋的距离转化为超级源点 S 到海洋的距离。

由于标识所用的数据结构是矩阵 mat ，所以我们同样用 `矩阵 d[x][y]` 来代表陆地 `(x, y)` 到海洋点的最短距离，那么答案就是 `d[][] 矩阵中最大的 d[x][y]` 。因为我们求得是陆地到海洋的最短距离，所以符合题目条件：找出一个海洋单元格，这个`海洋`单元格`到离它最近`的`陆地`单元格的距离是最大的。首先，我们要明确这样一个点：我们是将

方法三：动态规划

## 834. 树中距离之和
该题用暴力解法为 O(n^2) 如果用换根 DP 则只有 O(n) 的时间复杂度，直接上代码：

注意其中的`语法格式`：
1. 其中 Edge 的构造函数 `Edge()` 与 `Edge(int _y)` 必须要有，否则无法初始化，
2. 对于全局变量邻接表 G 来说，必须在函数中进行初始化，否则一定会出错

换根 DP 的关键：可以通过一个根节点 root1 的值 f(root1)，和它的性质(如 root1 的子树大小，或者它的深度等等性质)来更新出他的子树为根的结果 f(root2) 此时就可以使用换根 DP。

关键就在于这个换根 DP 运用的`是有根树`，而`不是无根树`，这就意味着它是`固定了一个根`的，所以才有子树之说，`否则每个节点都是一整颗树`，`就没有子树之说`了。这题我们利用到的性质是 x 的子树大小：对于每个节点 x，以他为根的子树大小为 size[x], 以他为节点的距离之和为 f[x], 那么对于它的一个孩子 y 来说，有： f[y] = f[x] + (n - size[y]) - size[y]; 其中 n - size[y] 代表`除了`以 y 为根的子树的其他所有节点的数量，size[y] 代表子树的大小，为什么是这样的状态转移方程呢？可以从`灵神的题解图`中容易得出，就是由于每个边长都是 1 ，所以这个`子树节点和`可以代表了距离之和。

附灵神题解：https://leetcode.cn/problems/sum-of-distances-in-tree/solutions/2345592/tu-jie-yi-zhang-tu-miao-dong-huan-gen-dp-6bgb/
```c++
struct Edge{
    int y;
    Edge() { y = 0; }
    Edge(int _y) { y = _y; }
};
vector<vector<Edge> > G;
vector<int> size, ans;
int n;
void insert(int x, int y) {
    G[x].push_back({y});
}
void DFS(int x, int fa, int depth) {
    ans[0] += depth;
    for (auto [y] : G[x]) {
        if (y == fa) continue;
        DFS(y, x, depth + 1);
        size[x] += size[y];
    }
}
void reroot(int x, int fa) {
    for (auto [y] : G[x]) {
        if (y == fa) continue;
        ans[y] = ans[x] + n - 2 * size[y];
        reroot(y, x);
    }
}
vector<int> sumOfDistancesInTree(int _n, vector<vector<int>>& edges) {
    n = _n;
    G = vector<vector<Edge> >(n);
    ans = vector<int>(n, 0);
    size = vector<int>(n, 1);
    for (int i = 0; i < edges.size(); i ++ ) {
        int x = edges[i][0], y = edges[i][1];
        insert(x, y), insert(y, x);
    }
    DFS(0, -1, 0);
    reroot(0, -1);
    return ans;

}
```

## 128. 最长连续序列
该题是 `枚举 + 哈希表`

注：该学的是枚举，而不是哈希表，枚举学不会，你暴力做法都做不出！！！

枚举的应该是：`以 x 为起点，最长连续序列`。我们用一个 哈希表 来存储 nums[] 中的所有元素，如果 `表中` 存在 x + 1 则继续找 x + 2, x + 3 ... 否则截止。

学会了如何枚举才能学会如何使用哈希表！！！，所以哈希表不是关键，关键是学习如何暴力枚举。

优化：我们如果 x - 1 存在，则我们不会枚举 x，而是跳过它，因为我肯定能在 nums[] 中遇到 x - 1, 到时候直接去枚举 x - 1 就是了，因为 x - 1 生成的答案一定会大于 x 生成的答案。

代码如下：
```c++
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> S;
    for (auto e : nums) S.insert(e);
    int ans = 0;
    for (auto e : nums) {
        if (S.count(e - 1)) continue;   // 如果存在 x - 1 则不枚举它
        int len = 1;        // 否则枚举以它为开头的这个序列
        while (S.count( ++ e)) len ++ ;     // 循环去找 x + 1, x + 2, x + 3, 直到不能继续下去
        ans = max(ans, len);
    }
    return ans;
}
```

## 62. 不同路径
我们不用 DP，而是采用组合来做：求从 n 个元素中选取 m 个元素的组合数量

```c++
// 方法 1：迭代
int C(n, m) {   // 时间复杂度为 O(n) 
    long long ans = 1;
    for (int x = n, y = 1; y <= min(m, n - m); x -- , y ++ ) {   // x 为分子，y 为分母
        ans = ans * x / y;      // 注意这里一定不能写成 ans *= x / y, 因为这样等价于 ans = ans * (x / y) ，而 x 不一定能整除 y ！！
    }
    cout << ans;
}

// 方法 2： 动态规划
// 该方法适用于你要 `多次` 使用 `不同的 n 和不同的 m` ，因为这是打出一个组合数的表
int C[n][n];
for (int i = 0; i <= n; i ++ )
    for (int j = 0; j <= i; j ++ )
        if (j == 0) C[i][j] = 1;
        else C[i][j] = C[i - 1][j] + C[i - 1][j - 1];

int C(int n, int m){
    return C[n][m];
}
```

## 713. 乘积小于 K 的子数组
本题为 `数学 + 滑动窗口`。此题类似于 `209. 长度最小的子数组` ，如何理解题意转化为滑动窗口题最为关键，因为逻辑和数学很重要。

本题还有点动态规划的味道：即我们着重考虑的是：`以 r 为结尾的子数组`

数学逻辑为：我们`固定`子数组的`右端点 r`，在此基础上找到`长度最长`且满足 `连乘 < k` 的这样一个子数组，那么此时的这样一个子数组能够具有 `r - l + 1` 个`不同的右端点为 r 的连续子数组`。

根据上述数学逻辑，我们很容易写出代码：

注意：由于可能存在子数组长度为 1 时也满足 `cur >= k` 的情况，所以我们要在 while 循环中添加 l <= r 这一条件，防止数组越界。

但我的建议是无论怎样都加上 l <= r 这一条，因为防止忘记，且该做法更为通用，加上更好
```c++
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
    int r = 0, l = 0, cur = 1, n = nums.size();
    int ans = 0;
    for (; r < n; r ++ ) {
        cur *= nums[r];
        while (cur >= k && l <= r) cur /= nums[l], l ++ ;   // 我的建议是无论怎样都加上 l <= r 这一条，可以防止遗忘，且该做法更为通用，加上更好
        if (cur < k) ans += r - l + 1;
    }
    return ans;
}
```

## 3. 无重复字符的最长子串
本题为 `滑动窗口 + 哈希表` 此题也和 `713. 乘积小于 K 的子数组`, `209. 长度最小的子数组` 类似，但是最和 `713. 乘积小于 K 的子数组` 类似，因为它们都类似于动态规划，是需要分析`固定以 r 为结尾`的窗口所具有的性质。

代码如下：
```c++
int lengthOfLongestSubstring(string s) {
    int S[256];     // 用数组代表哈希表
    memset(S, 0, sizeof S);
    int l = 0, r = 0, n = s.size(), ans = 0;
    for (; r < n; r ++ ) {
        S[s[r]] ++ ;
        while (S[s[r]] >= 2) S[s[l]] -- , l ++ ;    // 当固定的右端点 r 使得窗口内有重复元素出现，则将左端点像右移动
        ans = max(ans, r - l + 1);
    }
    return ans;
}
```

`总结`：实际上我们分析的这三个题都绕不过一个非常核心的理念：即固定右端点型滑动窗口，虽然叫做同向双指针型，但是其本质我觉得是`固定右端点型`滑动指针题目。即我们所需要的窗口是 `固定了右端点 r `所具有的一个`最优解`的窗口。本质上我觉得是动态规划的另一种实现方法，即`滑动窗口实现动态规划`。也就是去枚举所有子数组的右端点，而`状态就是`所维护的这个窗口，用这个右端点去构造一个最优窗口。术语：`以 r 为右端点(结尾)的所有子数组`。

## 525. 连续数组
此题为：`逻辑转化 + 前缀和 + 哈希表`

逻辑转化：我们将 0 看作 -1 ，然后计算每个位置 i 的前缀和，并记录下每个前缀和第一次出现时的下标，那么以后每次出现同样的前缀和就代表出现了 0 与 1 相同的字符串。
```c++
int findMaxLength(vector<int>& nums) {
    unordered_map<int, int> mp;
    int n = nums.size(), sum = 0;
    mp[0] = -1;
    int ans = 0;
    for (int i = 0; i < n; i ++ ) {
        sum += nums[i] ? 1 : -1;
        if (mp.count(sum)) ans = max(ans, i - mp[sum]);
        else mp[sum] = i;
    }
    return ans;
}
```

## 560. 和为 K 的子数组
这题`不用逻辑转化`，只需要`数学转化`即可。其数学转化及其证明已经出现在`刷题笔记`中，这里就不写了。这里主要是针对`前缀和与子数组`做个总结，并从另一种角度来分析该题目，即动态规划的角度分析。

总结：一旦有`子数组求和`问题，我们就要想到是否可以用前缀和来解决这个问题，因为前缀和可以用 O(1) 时间内求出任意区间内和。所以前缀和与子数组和是密切联系的。

从动态规划的角度分析：这题是`多窗口 + 动态规划`，很类似与`滑动窗口 + 动态规划`。我们只需要用哈希表`记录`下 `左端点 l 固定为 0` 右端点 i 变化的这样的窗口，使得其和等于 sum 的窗口`个数(记录在 mp 中)`(这里是固定了左端点的多窗口)。那么我们的右端点遍历到 i 时，如果存在和为 sum - k 前面所说这样的窗口，则就是 `ans += mp[sum - k]` 其中，mp[sum - k] 为这样的窗口的个数。

```c++
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> mp;
    mp[0] = 1;
    int ans = 0, sum = 0, n = nums.size();
    for (int i = 0; i < n; i ++ ) {
        sum += nums[i];
        if (mp.count(sum - k)) ans += mp[sum - k];
        mp[sum] ++ ;
    }
    return ans;
}
```

## 325. 和等于 k 的最长子数组长度
见刷题笔记

## 643. 子数组最大平均数 I
此题为长度固定型滑动指针，其中我们枚举右端点，而长度 len 是知道的，所以我们就知道左端点为 r - len + 1。通过这些理论知识我们就可以写出代码：

代码`实现技巧`：`两步走`
1. 通过窗口长度，首先生成一个初始窗口 [0 ... len - 1]
2. 再枚举这个窗口的右端点为 len 开始滑动。

```c++
double findMaxAverage(vector<int>& nums, int k) {
    int sum = 0, n = nums.size();
    for (int i = 0; i < k; i ++ ) sum += nums[i];
    int tmp = sum;
    for (int r = k; r < n; r ++ )
        sum += nums[r] - nums[r - k],
        tmp = max(tmp, sum);
    return 1.0 * tmp / k;
}
```

## 567. 字符串的排列
通过 `643. 子数组最大平均数 I` 这题，我们理解到了如何使用固定长度的滑动窗口，那么同样，我们这题也可以利用这种思想：

`前置知识`：如何判断`一个序列是否是另一个序列的排列`？
我们构建两个数组，cnt1[] 与 cnt2[]，其中 cnt 中记录了序列中的每个相同元素的数量。如果 cnt1[] 与 cnt2[] 完全相同，则说明它们是相互的排列。`注`：如果值域特别大，那么一定要使用 `unordered_map<int, int>` 否则会爆内存，而`小写字符集的值域为 26`，非常小，所以不用 unordered_map 了。


构建一个滑动窗口，并记录窗口内所有字符集的数量。然后`通过上面的分析`判断是否这个窗口是 s1 的排列即可。

代码`实现技巧`：`两步走`        ->   注：` 该模板已被推翻`，更好的模板在题目：`76. 最小覆盖子串` 中实现
1. 通过窗口长度，首先生成一个初始窗口 [0 ... len - 1]
2. 再枚举这个窗口的右端点为 len 开始滑动。

```c++
vector<int> cnt1, cnt2;
bool check() {
    for (int i = 0; i < 26; i ++ )
        if (cnt1[i] != cnt2[i]) return 0;
    return 1;
}
bool checkInclusion(string s1, string s2) {
    cnt1 = vector<int>(26);
    cnt2 = vector<int>(26);
    int n1 = s1.size(), n2 = s2.size();
    if (n1 > n2) return 0;
    
    for (int i = 0; i < n1; i ++ ) cnt1[s1[i] - 'a'] ++ ;   
    for (int i = 0; i < n1; i ++ ) cnt2[s2[i] - 'a'] ++ ;   // 生成滑动窗口
    if (check()) return 1;

    for (int i = n1; i < n2; i ++ ) {
        cnt2[s2[i] - 'a'] ++ , cnt2[s2[i - n1] - 'a'] -- ;
        if (check()) return 1;
    }
    return 0;
}
```

## 438. 找到字符串中所有字母异位词
此题与就是题目`567. 字符串的排列`的变形，非常类似，更本不需要多做解释，原理直接看 题目`567. 字符串的排列` 的分析即可

直接上代码：

注：改模板`已被推翻`，有更加好的模板在题目： `76. 最小覆盖子串`
```c++
vector<int> cnt1, cnt2;
bool check(){
    for (int i = 0; i < 26; i ++ )
        if (cnt1[i] != cnt2[i]) return 0;
    return 1;
}
vector<int> findAnagrams(string s, string p) {
    vector<int> res;
    cnt1 = vector<int>(26);
    cnt2 = vector<int>(26);
    int n1 = s.size(), n2 = p.size();
    if (n1 < n2) return res;
    for (int i = 0; i < n2; i ++ ) cnt1[s[i] - 'a'] ++ ;
    for (int i = 0; i < n2; i ++ ) cnt2[p[i] - 'a'] ++ ;
    if (check()) res.push_back(0);
    for (int i = n2; i < n1; i ++ ) {
        cnt1[s[i] - 'a'] ++ , cnt1[s[i - n2] - 'a'] -- ;
        if (check()) res.push_back(i - n2 + 1);
    }
    return res;
}
```
## 3. 无重复字符的最长子串
经过以上 `438. 找到字符串中所有字母异位词`, `567. 字符串的排列`, `643. 子数组最大平均数 I` 这三个题目之后，我们再回顾，且总结这一题。

关键点：`切入口`：我们发现，这是一个窗口区间长度需要改变的题目，即`固定右端点`，`左端点需要进行改变`，所以我们在做这种滑动窗口的题目时，切入口有四点，只要先把这四个切入点仔细思考好，才能得到最终正确的结果。这四点入下：

1. 固定长度 or 可变长度 ?
2. 固定端点 or 可变端点 ？

每个滑动窗口的题目都需要从这四点入手，其中 `838. 推多米诺` 是 `可变长度` 滑动窗口的一道典型题目，可以看下如何去用滑动窗口解决。

## 76. 最小覆盖子串
优化 cnt1 与 cnt2 组成的窗口性质，我们需要 `添加`一个更加好的数据结构，来优化让 cnt1[] 与 cnt2[] 每次都在 check 函数中逐个比较，该数据结构就是 `match`，即`当前窗口`成功匹配的字符数。

数据结构：`match(当前窗口内匹配成功的字符数)`。我们不断添加 match，直到满足 match == t.size() 即匹配成功的数量与 t 的大小相同，我们认为此时 s[l ... r] 能够完全覆盖 t，注意`此时我们将不再减少 match`，而是一直让 match 保持完全匹配成功的状态，即让当前窗口`一直处于`能够`完全匹配状态`

我们直接`推翻上述所有模板`，来分析一个更为通用的模板，首先，我们需要明确：在循环中，`窗口右端点`向`右移动`时会发生什么？设窗口的性质为 f
1. 我们需要将右端点的值添加入我们维护的窗口性质 f 中

然后我们去让左端点向右移动，当满足一个条件时，它应该不断循环向右移动，那么左端点向右移动后会发生什么？
1. 我们需要将左端点的值在 f 中删除。

我们结合代码去分析该题。

代码如下
```c++
const int INF = 0x3f3f3f3f;
string minWindow(string s, string t) {
    int cnt1[256] = {}, cnt2[256] = {};
    int l = 0, r = 0, match = 0, n = s.size(), ansL = 0, ansR = INF; // 首先初始化 ansR 为无穷大，代表这个字符串超级大

    for (auto e : t) cnt2[e] ++ ;       // 让 cnt2 的性质加上

    for (; r < n; r ++ ) {
        cnt1[s[r]] ++ ;         // r 右移后，将 s[r] 加入窗口性质 cnt1 中
        if (cnt2[s[r]] >= cnt1[s[r]]) match ++ ;
        while (l < r && cnt1[s[l]] > cnt2[s[l]]) cnt1[s[l]] -- , l ++ ;     // 在满足 match 不能降低的条件下(即维护窗口一直处于完全匹配状态)，将 s[l] 移除窗口性质 cnt1 中。
        if (match == t.size() && r - l < ansR - ansL)       // 如果完全匹配且能得到更优结果，则直接更新答案。
            ansL = l, ansR = r;
    }

    return ansR == INF ? "" : s.substr(ansL, ansR - ansL + 1);  // 如果从未更新过，则返回 "" ，否则返回结果字符串。
}
```

我们尝试利用这个 match 模板来解决固定长度的窗口问题：`438. 找到字符串中所有字母异位词`，并且`不用一次性`先直接建立一个固定长度的窗口，而是在 for 循环内判断这个窗口是否满足大小为 k 。

顺序方面的`技巧`：
1. 首先，由于是固定窗口，所以我们不需要使用 while 循环来建立一个窗口，而是利用 if(l < r - n1 + 1) 来判断是否需要 `左端点是否需要右移` 即可。
2. `在逻辑上对 r 来说`，由于每次都是先 r ++ 了，然后将 `s2[r]` 添加到性质中去。所以我们 for 循环内`一开始就要` cnt2[s2[r]] ++ 。但是`在逻辑上对于 l ` 来说，我们必须是`先`将 s2[l] 的性质`删除`，才能进行 l ++ , 而不是 l 先右移再添加性质，这个要特别注意

对 match 的说明：
1. 该字符串添加 s2[r] 后，如果有 `cnt1[s2[r]] >= cnt2[s2[r]]` 则说明能够`匹配` s1 中的一个字符，否则要么是无效匹配( s2 中不存在这样的字符)，要么是 `冗余匹配` (该窗口内该字符已经超过所需要匹配的字符) 。
2. 该字符串删除 s2[l] 后，如果有 `cnt2[s2[l]] < cnt1[s2[l]]` 则说明`会失配` s1 中的一个字符，否则要么不会产生失配(s2 中不存在这样的字符)，要么该字符还有冗余，并不需要删除可匹配项。


```c++
bool checkInclusion(string s1, string s2) {
    int match = 0, n1 = s1.size(), n2 = s2.size();
    if (n2 < n1) return 0;
    int cnt1[256] = {}, cnt2[256] = {};
    for (int i = 0; i < n1; i ++ ) cnt1[s1[i]] ++ ;
    int l = 0, r = 0;
    for (; r < n2; r ++ ) {
        cnt2[s2[r]] ++ ;
        if (cnt1[s2[r]] >= cnt2[s2[r]]) match ++ ;
        if (l < r - n1 + 1) {
            cnt2[s2[l]] -- ;
            if (cnt2[s2[l]] < cnt1[s2[l]]) match -- ;
            l ++ ;              // 一定是先移除性质 l 才能右移，不能先右移，先后顺序必须搞清
        }
        if (match == n1 && r = l + n1 - 1)  // 括号中的第二个判断可以删除，因为如果 match == n1 则后面那个窗口大小条件必然满足。
            return 1; 
    }
    return 0;
}
```


`总结`：关键点：
1. `性质添加删除`和`端点左移右移`之间的代码顺序问题
2. `何时`能算做匹配，`何时`才能算作失配


## 49. 字母异位词分组
初看这是一个滑动窗口题，但实际上它是一个 `分类 + 并查集思想 + 排序` 的一道综合题。
1. 分类思想：我们将同一种排列的字符串看作同一组，那么我们该如何定义它的特征标识呢？即我们需要通过该字符串知道它应该分到哪个组。
2. 排序：我们就是利用排序来实现识别功能，即定义它的特征标识，如果`按字典序重新排列(sort)`该字符串后相等的话，则说明它们是同一组，

代码如下：
```c++
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string> > mp;
    for (auto e : strs) {
        string key = e;
        sort(key.begin(), key.end());
        mp[key].push_back(e);
    }
    vector<vector<string> > res;
    for (auto [k, v] : mp) 
        res.push_back(v);
    
    return res;
}
```
## 953. 验证外星语词典  :LCR 034. 验证外星语词典
此题为 `哈希表 + 相对位置`

该题是一个非常有意思的题目，需要去`学习`这个思想：

首先，记 index[c] 代表 `字符 c` 在一个特定的字典(字典 order)(该字典为`字符的集合`)中的`相对位置`。那么我们可以`通过 order[] 这个字典`，来构建这个 `字符相对位置表 index[]`, 生成代码为： `index[order[i] - 'a'] = i` 也就是 `字符 order[i]` 的相对位置为 i。

`关键本质`：其实本质上也就是一个哈希表，让我们查找 `字符 ch` 在 order 字典中的位置时，不用去通过遍历这个 order[] 数组，而是直接记录下

生成了字符相对位置表，我们就要去比较两个单词的字典序。如何去比较字典序，我们直接在代码中进行`学习`：

```c++
bool isAlienSorted(vector<string>& words, string order) {
    int index[26];
    for (int i = 0; i < order.size(); i++) 
        index[order[i] - 'a'] = i;
    
    for (int i = 1; i < words.size(); i ++ ) {
        bool flag = 0;
        for (int j = 0; j < words[i - 1].size() && j < words[i].size(); j ++ ) {        // 记 words[i - 1] 为 s1, words[i] 为 s2.
            int pos1 = index[words[i - 1][j] - 'a'];        // s1[j] 在字典中的相对位置记为 pre
            int pos2 = index[words[i][j] - 'a'];            // s2[j] 在字典中的相对位置记为 cur
            if (pos1 < pos2) { flag = 1; break; }            // 如果能判断字典序大小则返回，如果 pos1 < pos2 则说明 s1 字典序较小
            else if (pos1 > pos2) return 0;                  // 如果能判断字典序大小则返回，如果 pos1 > pos2 则说明 s1 字典序较大
        }
        if (!flag && words[i - 1].size() > words[i].size()) return 0;   // 如果在两个字符串长度(取较小的)之内全部比较完后，仍然无法判断两个字符串的字典序，则比较两者的大小
    }
    return 1;
}
```

进阶：我们通过转换，将 `由特定字典 order 生成` 的相对位置表 index[] 与字符 ch 结合转换为 一个新的字符，其中该字符仍是以 a 开头，只不过这个 a 并不是`通用英文字符集` {a, b, c, d ... , x, y, z} 的第一个字符，而是 order 中的第一个字符，这样就可以 `使用 string 内重载的比较符号` 了，而`不需要手动实现字典序比较`。

通过`代码`来观察实现方法：

```c++
bool isAlienSorted(vector<string>& words, string order) {
    int index[26] = {0};
    for (int i = 0; i < order.size(); i ++ )
        index[order[i] - 'a'] = i + 'a';        // order 字典生成的新字符集

    for (int i = 0; i < words.size(); i ++ ) 
        for (int j = 0; j < words[i].size(); j ++ )
            words[i][j] = index[words[i][j] - 'a'];     // 转换为 order 字典的单词
    
    for (int i = 1; i < words.size(); i ++ )
        if (words[i - 1] > words[i]) return 0;  // 直接用 string 的重载操作符
    
    return 1;
}
```

## 318. 最大单词长度乘积
优化一：利用 `哈希表 + 字符集` 可通过 O(26) 的时间复杂度判断两个单词是否包含重复字母，可以通过，但时间复杂度仍然太高了(用 `数组实现哈希表` 可以通过，用 `unordered_set<int> 够呛` )，如何用 O(1) 的时间复杂度呢？

优化二：利用 `集合 + 位运算 + 字符集` 可以使用 mask1 与 mask2 和 `&` 操作这个位运算，利用 O(1) 的时间复杂度来判断两个不同的字符串是否拥有同一个字母

代码如下：
```c++
int maxProduct(vector<string>& words) {
    int n = words.size();
    int mask[n];
    memset(mask, 0, sizeof mask);
    int ans = 0;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < words[i].size();j ++ )
            mask[i] |= 1 << (words[i][j] - 'a');
    for (int i = 0; i < n; i ++ )
        for (int j = i + 1; j < n; j ++ ) {
            if ((mask[i] & mask[j]) == 0)
                ans = max(ans, (int)(words[i].size() * words[j].size()));
        }
    return ans;
}
```

## 680. 验证回文串 II
此题为 `贪心 + 分类讨论` ：

开始时，我们按照一般处理回文串的思路来处理该字符串，设置 l 与 r 指针，直到 `第一次` 遇到不匹配的位置(如果全部匹配则无需删除)。

那么我们就删除 l 或者 r 位置的字符串，然后再按一般思路来处理删除过后的字符串。

总结：`删除的常用处理方法是忽略`，而`不是`真正的`删除`，在人类视角中，删除很容易办到，但是在计算机中，`忽略比实际删除更容易办到`

代码简洁性技巧：我们创建一个函数判断`忽略一个字符后`是否能够满足回文串：

代码如下

该代码已被推翻，正确解法见：`LeetCode 1216. 验证回文字符串 III`
```c++
class Solution {
public:
    bool check(string &s, int l, int r) {
        for (; l < r; l ++ , r -- )
            if (s[l] != s[r]) return 0;
        return 1;
    }
    bool validPalindrome(string s) {
        int l = 0, r = s.size() - 1;
        for (; l < r; l ++ , r -- )
            if (s[l] != s[r]) break;
        return check(s, l + 1, r) || check(s, l, r - 1);
    }
};
```

## LeetCode 1216. 验证回文字符串 III
这题就是删除回文串的终极问题，其实我们又忘了一个转化，删除有两种转化：
1. 转化为保留
2. 转化为忽略

该题应该转化为 保留，而不是忽略。我们计算子序列中最长的回文子序列长度 len，然后判断 n - len 和 k 的大小即可
```c++
bool isValifalindrome(string s, int k) {
    int i, j, n = s.size();
    vector<vector<int>> f(n,vector<int>(n,0));
    for(i = 0; i < n; ++i) f[i][i] = 1;

    for(j = 0; j < n; ++j)
        for(i = j-1; i >= 0; --i)//区间从小往大，所以逆序
            if(s[i] == s[j])
                f[i][j] = f[i+1][j-1]+2;
            else
                f[i][j] = max(f[i+1][j], f[i][j-1]);
        
    
    return n-f[0][n-1] <= k;
}
```


## 100234. 在矩阵上写出字母 Y 所需的最少操作次数
该题是一个不错的`数学枚举`题，我们主要在此题积累两个知识：

1. 代码实现判断一个坐标是否属于 Y 的技巧：`利用数学坐标图来判断`
2. 如何通过`枚举`来`迭代`出`最少`改变次数？     // 这是利用枚举来思考这题的关键，我们构建 cntY 与 cntN(Cnt Not Y) `就是为了`这一步的`枚举迭代`

对于问题 1 ：
```c++
bool isY (int i, int j) {
    if (i - j == 0 || i + j == n - 1) return i <= n / 2;    // 利用数学坐标图来判断它是否在主或副对角线上 : x - y = c or x + y = d
    return i >= n / 2 && j == n / 2;
};
```
对于问题 2：

数学分析：
1. 正向思考：我们需要按照题目限制，将 Y 中的所有数字变为同一个数字，同时将 非Y 中的数字变为同一个数字，且前者和后者需要不同，那么我们可以枚举它们所变成的数字：i, j 其中 i != j 。那么我们就可以写出它们需要改变的数字。
2. `反向思考`：利用正向思考写代码有一个很不方便的处理：举例：即我们还需要去分析在 cntY 中，哪些数字不为 i，那么我们需要将他们加入该变量 change 中。这是比较麻烦的，可以用 `dirs[6][4] 数组` 来实现，具体方法可以看我 Leetcode 上第一次提交的代码。我们`反向来思考`这个问题：我们将无需改变的数字数目记录下来，那么答案最终为 n^2 - notChange 这样就`无需`去枚举哪些数字不为 i 了，非常方便。
3. `贪心 + 反向` 如果`不止` 0，1，2 `而是有更多的数字`，那么我们需要先对 `cnt排序` 然后选出 cntY 中最大值和 cntN 中最大值，同时要保证它们对应的数字不一样，如果一样则选择次大值即可。
代码：
```c++
int n;
bool isY(int i, int j) {
    if ((i - j == 0 || i + j == n - 1) && i < n / 2) return 1;      // 我们以后判断主对角线就用 i == j, 判断副对角线就用 i + j == C ，这是一元函数图像的应用，特别注意
    if (i >= n / 2 && j == n / 2) return 1;
    return 0;
}
int minimumOperationsToWriteY(vector<vector<int>>& mat) {
    n = mat.size();
    int cntY[3] = {0}, cntN[3] = {0};
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
            if (isY(i, j)) cntY[mat[i][j]] ++ ;
            else cntN[mat[i][j]] ++ ;

    int notChange = 0;
    
    for (int i = 0; i < 3; i ++ )
        for (int j = 0; j < 3; j ++ )
            if (i != j)
                notChange = max(notChange, cntY[i] + cntN[j]);
    return n * n - notChange;
}
```

## 539. 最小时间差
这是一道`逻辑分析`题，`需要`去`想到`：最小时间差一定是相邻元素才能构造出来，因此我们通过排序之后去寻找相邻元素之间的时间差即可！

我们可以先将时间转化为整数的形式然后排序，或者我们可以先将 string 类型(因为 string 的特性允许这样)的时间进行排序，然后再转化为整数进行比较。

`细节`：我们还要考虑最后一个时间`跨越 24:00 之后`距离第一个时间的差值，这是一个特例，注意这个细节。

代码实现：

```c++
int convert(string s) {
    int hour = (s[0] - '0') * 10 + (s[1] - '0');
    int min = (s[3] - '0') * 10 + (s[4] - '0');
    return hour * 60 + min;
}
int findMinDifference(vector<string>& vec) {
    int n = vec.size();
    sort(vec.begin(), vec.end());   // 将时间从小到大进行排序

    int tBegin = convert(vec[0]), tEnd = convert(vec.back());
    int ans = tBegin + 24 * 60 - tEnd;  // 初始化为最后一个时间跨越 24:00 到第一个时间的差值
    
    // 进行迭代出最小值
    for (int i = 1; i < n; i ++ ) {
        int t1 = convert(vec[i - 1]), t2 = convert(vec[i]);
        ans = min(ans, t2 - t1);
    }
    return ans;
}
```

利用库函数`更加优雅`：
```c++
int findMinDifference(vector<string>& words) {
    sort(words.begin(), words.end());
    int ans = 0x3f3f3f3f;
    for (int i = 0; i < words.size() - 1; i ++ ) {
        int h1, m1, h2, m2;
        cout << words[i] << endl;
        sscanf(words[i].c_str(), "%d:%d", &h1, &m1);
        sscanf(words[i + 1].c_str(), "%d:%d", &h2, &m2);
        int t1 = h1 * 60 + m1, t2 = h2 * 60 + m2;
        ans = min(ans, t2 - t1);
    }
    int a1, b1, a2, b2;
    sscanf(words[0].c_str(), "%d:%d", &a1, &b1);
    sscanf(words.back().c_str(), "%d:%d", &a2, &b2);
    ans = min(ans, 24 * 60 + a1 * 60 + b1 - (a2 * 60 + b2));
    return ans;
}
```

## 735. 小行星碰撞
这是一个非常好的栈的模拟题，我一开始以为这个题目和 `838. 推多米诺` 是类似的一个可以用双指针解决的一个问题，结果却不对。

因为这个不像 `推多米诺` 那样是要保留元素的，这是需要去删除元素的。那么我们分析过，删除元素最好的`模拟方法`是`忽略`，但是我们该如何实现忽略呢？这又是有难度的。经过分析后发现，我们应该去通过栈来模拟，因为碰撞后，我们可能一直要出栈，

代码技巧：为了将代码写得更加简洁，我们设置一个 flag 来判别栈外的那个行星是否爆炸。如果爆炸了就不用入栈，否则入栈。这将会对判断`是否将该次栈外行星`的思考量和代码复杂度降低。

代码如下：

```c++
vector<int> asteroidCollision(vector<int>& nums) {
    vector<int> stk;
    int n = nums.size();
    for (int i = 0; i < n; i ++ ) {
        bool flag = 1;      // 该栈外行星 nums[i] 依然存在 
        while (stk.size() && flag && nums[i] < 0 && stk.back() > 0) {   // 如果没爆炸就直到将栈中所有行星炸空为止
            flag = -nums[i] > stk.back();
            if (stk.back() <= -nums[i]) stk.pop_back();
        }
        if (flag) stk.push_back(nums[i]);
    }
    return stk;
}
```

## 739. 每日温度

经典的单调栈问题，但就是想不出要用单调栈，主要是因为会用暴力做法，然后不常用单调栈

我们在这个题目中`详细分析一下`：递增栈与递减栈的脑海想象图：

首先，我们在脑海中想象这样一张图像： 有一个`垂直的栈`，栈的`外面有一个元素`，而该元素就是当前遍历到的 nums[i], 然后 x(坐标轴) 的顺序是从栈底到栈顶，y(值域) 的顺序 `用递增递减来描述`。结论：如果`栈是单调递减`，则`栈经过一系列的 pop() 操作后`，栈外的元素一定要小于栈顶，否则该栈外元素 nums[i] 无法入栈。同理，如果栈是单调递增，则经过一系列 pop() 操作后，栈外的元素一定大于栈顶

代码：
```c++
vector<int> dailyTemperatures(vector<int>& nums) {
    int n = nums.size();
    vector<int> stk, res(n, 0);         // 代码技巧：先将 res 初始化为 n 个长度的，特殊状态 0
    for (int i = n - 1; i >= 0; i -- ) {
        while (stk.size() && nums[i] >= nums[stk.back()])
            stk.pop_back();
        if (stk.size()) res[i] = stk.back() - i;        // 如果栈中有元素，则更新为一般状态
        stk.push_back(i);
    }
    return res;
}
```

## LCR 041. 数据流中的移动平均值
该题为经典的队列应用题，因为我们要维护一个长度固定 <= size 的窗口，那么我们可以用队列来维护该床口，在实现时，可以使用 STL 的队列，或者自己 `用 l 与 r + 前缀和` 来维护这个队列。记住，队列存的是值，不是下标，我们对不同问题，存值还是存下标是具体情况具体分析的。

```c++
// 实现一：用 l 与 r 来维护该窗口
int s[10010];   
int k, id;
MovingAverage(int size) {
    k = size;
    memset(s, 0, sizeof s);
    id = 0;
}

double next(int val) {
    s[ ++ id] = val + s[id];
    int l = max(0, id - k);
    return 1.0 * (s[id] - s[l]) / (id - l);
}


// 实现二：

int k, sum;
queue<int> q;
MovingAverage(int size) {
    k = size, sum = 0;
}

double next(int val) {
    if (q.size() == k) sum -= q.front(), q.pop();   // 先出队，再入队。
    q.push(val), sum += val;
    return 1.0 * sum / q.size();
}

```

## 933. 最近的请求次数
这也是一道很好的队列应用题，其中，我们应该用 queue 存储`下标`即时间戳，`而不是值`。这如同单调队列的形式
```c++
// 实现 1： 利用 l 与 r 维护窗口
int l, r;
int q[10010];
RecentCounter() {
    l = 0, r = -1;
}

int ping(int t) {
    r ++ , q[r] = t;
    while (q[l] < t - 3000 && l < r) l ++ ;
    return r - l + 1;
}

// 利用队列维护窗口
queue<int> q;
RecentCounter() {
    // q 没有 clear() 方法，所以没法使用 q.clear() 来初始化，最多用 while (q.size()) q.pop(); 来清空
}

int ping(int t) {
    q.push(t);
    while (q.front() < t - 3000) q.pop();
    return q.size();
}
```

## acwing: 3498. 日期差值
经典日期模拟题，我们用 a[] 来记录每个月份的天数，并用 while() 循环来迭代出 `月份能累加出的天数` 和 `年份能累加出的天数`。我们的天数从第 0 天开始计算，然后我们直接返回 day(x) - day(y) 

代码：
```c++
#include<bits/stdc++.h>
using namespace std;
int a[13]={0,31,28,31,30,31,30,31,31,30,31,30,31};      //把每月的天数求出来，这里让 a[0] 等于 0 最为致命，因为我们是先 -- m ，当前月份的天数一定不能算上

bool isLeap(int y) {    // 这里是判断闰年，将他包装成一个函数，增强代码简洁性
    if (y % 4 == 0 && y % 100 !=0 || y % 400 == 0) return 1;
    return 0;
}

int day(int n) {
    int y = n / 10000;          //年
    int m = n % 10000 / 100;    //月
    int d = n % 100;            //日
    
    a[2] = isLeap(y) ? 29 : 28;     //判断，并赋值给 a[2] 
    
    while ( -- m) d += a[m];       //从上一月开始，逐步累加天数
    
    while ( -- y) d += isLeap(y) ? 366 : 365;       // 从上一年开始，逐步累加年数
    
    return d;       //返回一共的天数
}
int main()
{
    int x, y;
    while(cin >> x >> y)
        cout << abs(day(x) - day(y)) + 1 <<endl;    //由题目要求所示：如果两个日期是连续的我们规定他们之间的天数为两天，因此我们加一
}
```

## 172. 阶乘后的零
数学：我们统计从 1 ~ n 中计算出所有的数的因子为 5 的个数。


优化：我们利用技巧可不用枚举 1 ~ n, 直接算出 n! 的因子为 5 的个数，具体方法如下：
```c++
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n, ans = 0;
    cin >> n;
    while (n) {
        ans += n / 5;
        n /= 5;
    }
    cout << ans;
    return 0;
}
```

## AcWing 3497. 质数
利用线性筛法模板解题：

```c++
int p[n], cnt = 0;
for (int i = 2; i <= n; i ++ ) {
    if (!v[i]) p[ ++ cnt] = i;
    for (int j = 1; p[j] <= n / i; j ++ ) {
        v[p[j] * i] = 1;
        if (i % p[j] == 0) break;
    }
}
```

## 3512. 最短距离总和
使用 `floyd 算法 + 逻辑转换 + 反向思考` 实现

1. 逻辑转换：我们用忽略逻辑来代替点的删除，只去对那些`存留下来的点`进行边长的累加

2. 反向思考：我们将删除点看为逐个去增加点，并且只对`已经添上的点`(对应于`删除后存留`下的点)进行边长的累加。

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e3 + 10;
int G[N][N];

int main(){
    int n; cin >> n;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            cin >> G[i][j];
    int ans = 0;
    for (int k = n; k > 1; k -- )           // 反向枚举来进行更新所有的点。
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ ) {
                if (G[i][j] > G[i][k] + G[k][j]) 
                    G[i][j] = G[i][k] + G[k][j];
                if (i >= k && j >= k)       // 已经被添加入的点为 [k...n] 即对应于删除之后存留下的点
                    ans += G[i][j];
            }
    cout << ans;
    return 0;
}
```

## Acwing: 3478. 旧帐单
这是一个经典的`枚举模拟`题，我们利用枚举的方法来判断是否存在一个数字满足账单，并且数字需要从大到小枚举：

```c++
#include <bits/stdc++.h>
using namespace std;
int n, x, y, z;
int main(){
    while (cin >> n) {
        cin >> x >> y >> z;
        int flag = 0;
        for (int i = 9; i > 0; i -- ) {
            for (int j = 9; j >= 0; j --) {
                int sum = i * 10'000 + x * 1000 + y * 100 + z * 10 + j;
                if (sum % n == 0) {
                    printf("%d %d %d\n", i, j, sum / n);
                    flag = 1;
                    break;
                }
            }
            if (flag) break;
        }
            
        if (!flag) cout << 0 << endl;
    }
}
```

## 3486. 前序和后序
这个题目需要现学：它是 m 叉树，即如果你知道了子树的个数 cnt，那么它有 C m 取 cnt 个格子放置子树，然后依次放置`子树1，子树2，...，子树cnt`。`特别注意`，在`格子内部的子树不能排序`，即一定是从子树 1 一直到子树 cnt。即我们只能组合，不能排列。因为前序遍历一个 m 叉树时，其子树的相对顺序必须肯定是固定下来的。

如何分割子树：

1. 根节点一定是 arr1[0], 所以子树我们从 arr1[1] 开始寻找
2. 我们依次找出子树 1，子树2，子树3，然后递归它们，求出 sum 即子树的不同组合方式，然后乘到 sum 中，代表当前这颗树的总组合方式。

```c++
#include <bits/stdc++.h>
using namespace std;
int n, C[30][30];
string arr1, arr2;
int DFS(int l1, int r1, int l2, int r2) {
    int sum = 1, cnt = 0;
    int start1 = l1 + 1, start2 = l2, len = 1;
    for (; start1 <= r1; ) {
        while (arr1[start1] != arr2[start2 + len - 1]) len ++ ;
        sum *= DFS(start1, start1 + len - 1, start2, start2 + len - 1);     // 这里的组合方案一定是乘法，而不是加法
        cnt ++ ;
        start1 += len, start2 += len; len = 1;
    }
    return sum * C[n][cnt];
}


int main(){
    memset(C, 0, sizeof C);
    for (int i = 0; i <= 26; i ++ )
        for (int j = 0; j <= i; j ++ ) 
            if (j == 0) C[i][j] = 1;
            else C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
            
    while (cin >> n) {
        cin >> arr1 >> arr2;
        int n = arr1.size();
        cout << DFS(0, n - 1, 0, n - 1) << endl;
        
    }
    return 0;
}
```

## AcWing 3479. 数字反转

我们可以用 `stringstream库函数` 来写此题，但是这里介绍`通常的写法`：

该题需要我们设计一个数字翻转函数：
```c++
#include <bits/stdc++.h>
using namespace std;
int reverse(int x) {
    int ans = 0;
    while (x) {
        ans = ans * 10 + x % 10;
        x /= 10;
    }
    return ans;
}
int main() {
    int a, b;
    while(cin >> a >> b){
        if (reverse(a) + reverse(b) == reverse(a + b)) cout << a + b;
        else cout << "NO";
        cout << endl;
    }
    return 0;
}
```

## 1976. 到达目的地的方案数
`动态规划 + dijsktra`

状态定义：定义 `f[i]` 表示节点 0 到节点 i 的最短路个数。

在用 d[x] 更新 dis[y] 时:
1. 如果 d[x]+ g[x][y] < d[y]，说明从 0 到 x 再到 y 的路径是目前最短的，所以更新 f[y] 为 f[x]。
2. 如果 d[x]+ g[x][y] == d[y]，说明从 0 到 x 再到 y 的路径与之前找到的路径一样短，所以把 f[y] 增加 f[x]。
3. 初始值: f[0] = 1，因为 0 到 0 只有一种方案，即原地不动。
答案: f[n - 1]。

思考：如果使用 SPFA 呢？如果`直接在 SPFA 的更新边长上做动态规划`，则会`出错`！！

答：通过 `反向建边` + `在 d[] 数组上` `进行记忆化搜索`
答案为 DFS(n - 1) 。`特别注意：因为这是无向图`，所以是可以从 n - 1 这个节点往回搜索的，但是如果是`有向图呢`？那么我们就要额外创建一个 GR (graph reverse)矩阵，代表将有向图中所有的边都翻转过来简历的一个图。那么当求出全部的 d[] 后，此时就可以从 n - 1 节点开始往回进行记忆化搜索了。
```c++
int DFS(int x) {
    if (f[x] != -1) return f[x];

    LL &res = f[x]; res = 0;
    for (auto [y, z] : GR[x]) 
        if (d[x] == d[y] + z) 
            res = res + DFS(y);     // 这里一定不是  + f[y] , 而一定是 + DFS(y)     否则无法进行记忆化搜索！！！

    return res;
}
memset(f, -1, sizeof f);
f[src] = 1;           // 将源点 src 的路径数初始化为 1
for (int i = n - 1; i >= 0; i -- )      // 我们可以求出每一点的从源点到达该点的最短路径数，就是 DFS(i)
    DFS(i);
```

Dij 的算法：
```c++
const int INF = 0x3f3f3f3f3f3f3f3f;
const int N = 2e2 + 10;
const int MOD = 1e9 + 7;
typedef long long LL;
typedef pair<LL, LL> PII;
class Solution {
public:
    struct Edge{
        LL y, z;
        Edge() {}
        Edge(LL y, LL z) : y(y), z(z) {}
    };
    vector<Edge> G[N];
    void insert(LL x, LL y, LL z) {
        G[x].push_back({y, z});
    }
    LL d[N], f[N];
    int v[N];
    void Dij(int s) {
        memset(v, 0, sizeof v);
        memset(f, 0, sizeof f);
        memset(d, 0x3f, sizeof d);
        priority_queue<PII, vector<PII>, greater<PII>> q;
        q.push({0, s}), f[s] = 1;
        while (q.size()) {
            auto [dis, x] = q.top(); q.pop();
            if (v[x]) continue; v[x] = 1;
            for (auto [y, z] : G[x]) 
                if (d[y] > dis + z) d[y] = dis + z, q.push({d[y], y}), f[y] = f[x];
                else if (d[y] == dis + z) f[y] = (f[y] + f[x]) % MOD;
        }
    }
    int countPaths(int n, vector<vector<int>>& roads) {
        for (auto e : roads) {
            LL x = e[0], y = e[1], z = e[2];
            insert(x, y, z), insert(y, x, z);
        }
        Dij(0);
        return f[n - 1];
    }
};
```

## 洛谷P1194 买礼物

此题为最小生成树的改版问题，由于是`给出邻接矩阵的形式`，说明这是稠密图，则用 Prim 算法来实现：

`关键处理部分`：根据题目要求，我们需要将`矩阵中为 0 或者超过 A 的边值都赋值为 A`。因为不能优惠打折的必须用 A 的价格购买，而如果矩阵中的 边的价格超过了 A 则用 A 更为划算。

`初始化处理部分`：根据题目要求，购买第一件商品一定要花 A 元，所以先将 A 元添加入

`易错点`：一定是 p[px] = py 。 而不是 p[x] = py 这个一定要记住!!
代码：
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e3 + 10;
int G[N][N];
int v[N], d[N];
int A, n;
int res = 0;
void prim(int s){
    memset(d, 0x3f, sizeof d);
    memset(v, 0, sizeof v);
    d[s] = 0;       // 相当于讲 s -> s 这条边加入，等价于不加边 
    for (int i = 0; i < n; i ++ ) {
        int x = -1;
        for (int j = 0; j < n; j ++ ) 
            if (!v[j] && (x == -1 || d[j] < d[x])) x = j;
        v[x] = 1, res += d[x];      // 我们一共会添加 n 次边，其中第一次加的边为 d[s] = 0, 相当于每有添加该边，只添加了 n - 1 条边
        for (int y = 0; y < n; y ++ )
            if (!v[y]) d[y] = min(d[y], G[x][y]);
    }
}
int main() {
    cin >> A >> n;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ ) {
            int t; cin >> t;
            t = (t == 0 || t > A) ? A : t;      // 特别的，如果输入的 t == 0，那么表示这两样东西之间不会导致优惠
            G[i][j] = G[j][i] = t;
        }
    res = 0;
    
    prim(0);
    cout << A + res;    // 根据题目要求，购买第一件商品一定要花 A 元，所以先将 A 元添加入
    return 0;
}
```
## P1396 营救

该题目求的不是最短路径，而是要求`选的每条边`尽可能的短。举例：边 [1, 2, 1] 是要`优于` [3] 的，即使`路径长度 1 + 2 + 1 > 3 `，但是它的边的最大值最小。所以前者更符合题意。

那么我们就利用 kruskal 算法。

`返回时处理`：`当 s 与 t 点属于同一集合时`，我们找到了一个最优解，此时 return

代码：
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
const int INF = 0x3f3f3f3f;
struct Edge{
    int x, y, z;
    bool operator < (const Edge &other) const  {
        return z < other.z;
    }
};
Edge edges[N];
int p[N];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
int n, m, s, t, ans = 0;
void kru(){
    for (int i = 1; i <= n; i ++ ) p[i] = i;
    sort(edges, edges + m);
    for (int i = 0; i < m; i ++ ) {
        auto [x, y, z] = edges[i];
        int px = find(x), py = find(y);
        if (px == py) continue;
        ans = max(ans, z), p[px] = py;
        if (find(s) == find(t)) return;
    }
}
int main(){
    cin >> n >> m >> s >> t;
    for (int i = 0; i < m; i ++ ) {
        int x, y, z; cin >> x >> y >> z;
        edges[i] = {x, y, z};
    }
    kru();
    cout << ans;
    return 0;
}
```

## codeforces B. AND Sequences
网址：https://codeforces.com/problemset/problem/1513/B

此题为 `排列组合数学 + 位运算数学 + 逻辑分析`

题意：

输入 T(≤1e4) 表示 T 组数据。所有数据的 n 之和 ≤2e5。
每组数据输入 n(2 ≤ n ≤ 2e5) 和长为 n 的数组 a(0≤a[i]≤1e9)。

把数组 a 重新排列为数组 b，使得`对于所有 1≤ i ≤ n - 1`，满足：
b 的长为 i 的前缀的 AND，等于 b 的长为 n-i 的后缀的 AND（按位与）。

输出有多少个【元素下标不同】的 b，`模 1e9 + 7`。
例如 a=[1,1,1]` 有 6 个【元素下标不同】的排列`，虽然这些排列都是 [1,1,1]。

关键：`对于所有 1≤ i ≤ n - 1`，为了使得所有的这些 i 都满足这种性质，那么

设 a 的`所有元素的 AND 为 x`，我们`必须在最左和最右放 x`。
设 x 在 a 中出现了 cnt 次，那么有 A(cnt,2) = cnt * (cnt-1) 种放置两个 x 的方案。
剩下的 n-2 个数随意排，有 (n-2)! 种方案。

所以答案为 cnt*(cnt-1)*(n-2)!。

记得取模，以及使用 64 位 int。

代码如下：

细节：由于我们需要的是所有 & 运算的总和，所以我们必须初始化从全 1 即 0xffffffff 开始进行 &，且在有符号整数中 0xffffffff == -1
```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
const int MOD = 1e9 + 7;
int T, n;
int main() {
    cin >> T;
    while (T -- ) {
        cin >> n;
        vector<LL> nums(n);
        LL x = -1;      
        for (int i = 0; i < n; i ++ ) {
            cin >> nums[i];
            x &= nums[i];
        }
        
        LL cnt = 0;
        for (int i = 0; i < n; i ++ ) 
            if (nums[i] == x) cnt ++ ;
        
        LL ans = cnt * (cnt - 1);
        for (int i = 2; i <= n - 2; i ++ )
            ans = (ans * i) % MOD;
        
        cout << ans << endl;
    }
    return 0;
}
```

##  P1195 口袋的天空
此题是一个最小生成树的变形题，我们的`目标就是选取 n - k 条边`，来获得一个`生成森林`，如果无法选择这么多边，则无法生成，返回 No Answer。

那么我们可以用最小生成树的办法，保证了这个最小生成森林的边权和是最小的。

代码如下：

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
struct Edge{
    int x, y, z;
    bool operator < (const Edge &other) const {
        return z < other.z;
    }
};
Edge edges[N];
int n, m, k, res = 0;
int p[N];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
bool kru() {
    for (int i = 1; i <= n; i ++ ) p[i] = i;
    int cnt = 0;
    res = 0;
    sort(edges, edges + m);
    for (int i = 0; i < m; i ++ ) {
        auto [x, y, z] = edges[i];
        int px = find(x), py = find(y);
        if (px == py) continue;
        cnt ++ , res += z, p[px] = py;
        if (cnt == n - k) return 1;         // 生成 Ｋ　个生成森林，需要有　n - k　个边
    }
    return 0;
}
int main(){
    cin >> n >> m >> k;
    for (int i = 0; i < m; i ++ ) {
        int x, y, z; cin >> x >> y >> z;
        edges[i] = {x, y, z};
    }
    if (kru()) printf("%d", res);
    else printf("No Answer");
    return 0;
}
```

## 1462. 课程表 IV
此题为：`拓扑排序 + 动态规划` 或者 `floyd + 动态规划`

定义：`f[x][y]` 为 0 则 x 不是 y 的先决条件，为 1 则 x 是 y 的先决条件，那么如何进行状态转移呢？

对于每个 x 在`删除它的出边`，即进行`更新 ind[y] 时` ，`首先`让 f[x][y] = 1, `然后`我们遍历所有的节点 k，有转移方程： `f[k][y] = f[k][y] | f[x][y]`

那么我们就可以写出代码：
```c++
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
int ind[N];
void insert(int x, int y) {
    G[x].push_back({y}), ind[y] ++ ;
}
vector<bool> checkIfPrerequisite(int n, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
    bool f[n][n];
    memset(f, 0, sizeof f);
    memset(ind, 0, sizeof ind);
    for (auto e : prerequisites) {
        int x = e[0], y = e[1];
        insert(x, y);
    }
    queue<int> q;
    for (int i = 0; i < n; i ++ )   
        if (ind[i] == 0) q.push(i);
    while (q.size()) {
        for (int i = q.size(); i; i -- ) {
            int x = q.front(); q.pop();
            for (auto [y] : G[x]) {
                f[x][y] = 1;        
                for (int k = 0; k < n; k ++ )
                    f[k][y] |= f[k][x];
                
                if ( -- ind[y] == 0) q.push(y);
            }
        }
    }
    vector<bool> ans;
    for (auto e : queries) {
        int x = e[0], y = e[1];
        ans.push_back(f[x][y]);
    }
    return ans;
}
```

floyd 算法：floyd 可以计算所有的节点之间是否存在最短路径。那么我们运用到这里同样可以算出是否存在这样一条路径：
```c++

class Solution {
public:
    vector<bool> checkIfPrerequisite(int n, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        bool f[n][n];
        memset(f, 0, sizeof f);
        for (auto e : prerequisites) {
            int x = e[0], y = e[1];
            f[x][y] = 1;
        }

        for (int k = 0; k < n; k ++ )
            for (int x = 0; x < n; x ++ )
                for (int y = 0; y < n; y ++ ) 
                    f[x][y] |= (f[x][k] && f[k][y]);    // 注意这里的转移方程是 f[x][k] && f[k][y], 而不是 || 。因为需要 x -> k 与 k -> x 都有路径
        
        vector<bool> ans;
        for (auto e : queries) {
            int x = e[0], y = e[1];
            ans.push_back(f[x][y]);
        }
        return ans;
            
    }
};
```

## 310. 最小高度树
该题为`拓扑排序 + 规律`

对于任意一棵树来说，我们要`从叶子节点开始往内搜索`，一层一层的往内搜索，`类似于多源 BFS`。我们用拓扑排序来实现这一过程：因为叶子节点为入度为 1 的节点，所以我们将 `入度为 1 的节点` 入队，然后逐个`删除这些点依附的边`，最后只可能`剩下 1 个节点或者 2 个节点`(证明略)，那么此时就将最后余下的节点存入答案即可。

代码：

```c++
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
int ind[N];
void insert(int x, int y) {
    G[x].push_back(y); ind[y] ++ ;
}
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1) return {0};
    
    for (auto e : edges) {
        int x = e[0], y = e[1];
        insert(x, y), insert(y, x);
    }

    queue<int> q;
    
    for (int i = 0; i < n; i ++ )
        if (ind[i] == 1) q.push(i);
    int remain = n;
    while (remain > 2) {
        remain -= q.size();
        for (int i = q.size(); i; i -- ) {
            int x = q.front(); q.pop();
            for (auto [y] : G[x]) 
                if( -- ind[y] == 1) q.push(y);
            
        }
    }
    vector<int> ans;
    while (q.size())
        ans.push_back(q.front()), q.pop();
    
    return ans;
}
```

## 2115. 从给定原材料中找到所有可以做出的菜
这题是一道很有趣的拓扑排序问题，`关键难点`在于如何`构造`出这样一个符合题意的`有向无环图`？

答：用 unordered_map 来进行构造： `图 G` 和 `入度数组ind`：
1. 图 G 的签名：`unordered_map<string, vector<string>> G` 其中 G[s] 代表了 s 的出边链(对应于邻接表的`出边链表`)
2. 入度数组 ind 的签名为：`unordered_map<string, int>` 其中 ind[s] 代表了节点 s 的入度

我一开始不理解，觉得这样只能让`原材料`指向`菜`，但是缺少了菜指向原材料的边，直到我通过这个代码调试之后发现，`原材料中也含有菜`！!并不是说原材料就没有菜，所以调试代码很重要：
```c++
for (auto [x, y] : G) {     // 其中 x 为节点，y 为 边链表
    cout << x << ": ";      // 节点
    for (auto e : y) cout << e << " ";      // 边
    cout << endl;
}
```
而`对于 supplies 来说`，就是我们入度为 0 的节点，直接入队即可

代码实现如下：
```c++
vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
        unordered_map<string, vector<string> > G;
        unordered_map<string, int> ind;

        for (int i = 0; i < recipes.size(); i ++ ) {
            for (auto x : ingredients[i])
                G[x].push_back(recipes[i]);
            ind[recipes[i]] = ingredients[i].size();
        }

        vector<string> ans;
        queue<string> q;
        for (auto e : supplies)
            q.push(e);
        
        while (q.size()) {
            string x = q.front(); q.pop();
            for (auto y : G[x]) {
                if ( -- ind[y] == 0)
                    q.push(y), ans.push_back(y);
            }
        }
        return ans;
    }
```

## 2917. 找出数组中的 K-or 值
该题是一个`位运算 + 枚举`的题目，读清楚题意最为重要。

`题目的意思`：如果 nums[] 中所有数字中第 i 位为 1 的数字数目不小于 k 个，则我们将 (1 << i) 用 `或符号 |` 添加入答案中。

只有读懂了题意才能写出代码：
```c++
int findKOr(vector<int>& nums, int k) {
    int ans = 0;
    for (int i = 0; i < 31; i ++ ) {
        int t = 1 << i, cnt = 0;
        for (auto e : nums) 
            if (e & t) cnt ++ ;
        if (cnt >= k) ans |= 1 << i;
    }
    return ans;
}
```

## 洛谷 P3916 图的遍历
此题为： `倒序DFS + 反向建边`

首先，O(n^2) 绝对会超时，所以我们应该去思考怎么使用 O(n) 的复杂度呢？注意到我们 `正向DFS + DP `是行不通的，因为在我们进行正向 DFS 时，这个 v[x] 会被置为 1，那么就永远不会往回搜索了，但实际情况是可能需要往回搜索的，比如当`前节点 cur` 需要通过`刚才搜索过来的节点 pre `进行往回搜索才能得到更大的节点编号，而如果 v[pre] 已经置为 1 了，则无法进行反向搜索了。

那么我们考虑进行 `倒序DFS + 反向建边`，倒序 DFS 指的是我们应该从`节点编号大`的那个节点`开始`进行搜索，这样就可以让编号小的节点直接获得最大的节点的编号。反向建边则代表让最大编号的节点来搜索我们，而不是等我们去搜索它们来得到最大可能的编号，这样就可以避免了 O(n^2) 的搜索。

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
int v[N], f[N];
int n, m;
void insert(int x, int y) {
    G[x].push_back(y);
}
void DFS(int x, int MaxN) {
    if (v[x]) return ; v[x] = 1;
    f[x] = MaxN;
    for (auto [y] : G[x]) {
        if (v[y]) continue;
        DFS(y, MaxN);
    }
}
int main(){
    memset(v, 0, sizeof v);
    cin >> n >> m;
    
    for (int i = 0; i < m; i ++ ) {
        int x, y; cin >> x >> y;
        insert(y, x);       // 反向建边
    }
    for (int i = n; i >= 1; i -- )      // 倒序遍历
        DFS(i, i);
    for (int i = 1; i <= n; i ++ )
        printf("%d ", f[i]);
    return 0;
}
```

## 洛谷：P1113 杂务

此题就是一个拓扑排序问题，在处理这个问题时，我们可以建立一个 `编号为 n + 1 的超级汇点` 代表最后一个事件发生。或者也可以在最后处理答案时做一个巧妙的处理：ans = max(ans, ve[x] + mp[x]) 也就是当前最早开始时间还需要添加完成自己这个任务(活动)的时间。

在代码中我们`将所有的边先存储再处理`，实际上题目`限制`了 k 这个任务只能由 1 ~ k - 1 这些任务转移过来，所以`我们可以不用先进行存储`。但是为了题目的一般适用性，我们这里可以用更通用的办法进行处理。

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
const int INF = 0x3f3f3f3f;
typedef pair<int, int> PII;
struct Edge{
    int y, z;
    Edge() {}
    Edge(int y, int z) : y(y), z(z) {}
};
vector<Edge> G[N];
int ind[N], ve[N];
int n;
void insert(int x, int y, int z) {
    G[x].push_back({y, z}), ind[y] ++ ;
}
void topsort(){
    memset(ve, 0, sizeof ve);
    queue<int> q;
    for (int i = 1; i <= n; i ++ )
        if (ind[i] == 0) q.push(i);
    while (q.size()) {
        for (int i = q.size(); i ; i -- ) {
            int x = q.front(); q.pop();
            for (auto [y, z] : G[x]) {
                ve[y] = max(ve[y], ve[x] + z);
                if ( -- ind[y] == 0) q.push(y);
            }
        }
    }
}
int main() {
    memset(ind, 0, sizeof ind);
    unordered_map<int, int> mp;
    vector<PII> edges;
    cin >> n;
    for (int i = 1; i <= n; i ++ ) {
        int x, y, t; cin >> x >> mp[x] >> t;
        while (t) {
            edges.push_back({t, x});
            cin >> t;
        }
    }
    for (auto [x, y] : edges) {
        insert(x, y, mp[x]);
    }
    topsort();
    int ans = 0;
    for (int i = 1; i <= n; i ++ ) ans = max(ans, ve[i] + mp[i]);
    cout << ans ;
    return 0;
}
```

## 洛谷：P4017 最大食物链计数
此题和 `1976. 到达目的地的方案数` 一样，为 `路径 DP` 问题

那么我们定义 f[x] 为以 x 为终点的食物链路径数量，那么转移方程为：{ `f[y] = f[y] + f[x]` ，其中存在 x 指向 y 的边：x -> y }。

那么最终答案为：`if (outd[x] == 0) ans += f[x]`:这句代码代表了如果出度为 0，也就是说如果该节点为食物链最顶端的捕食者，那么就要让答案加上它。因为题目要求的是 ：`食物网中最大食物链的数量`， 所谓`最大食物链`就是从 ind 为 0 的点到 outd 为 0 的点的这一条路径，所以我们再 outd[x] == 0 时收集答案。

与 Dij `求最短路径条数` 的关键区别是：dij 要判断距离的 >= 和 ==，而此题不需要判断.

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
const int MOD = 80112002;
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
int ind[N], f[N], otd[N];
int n, m;
void insert(int x, int y) {
    G[x].push_back({y}), ind[y] ++ , otd[x] ++ ;    // 记录 ind[y] 的同时，记录 outd[x]
}
void topsort(){
    queue<int> q;
    memset(f, 0, sizeof f);
    for (int i = 1; i <= n; i ++ )
        if (ind[i] == 0) q.push(i), f[i] = 1;
    while (q.size()) {
        int x = q.front(); q.pop();
        for (auto [y] : G[x]) {
            f[y] = (f[y] + f[x]) % MOD;
            if ( -- ind[y] == 0) q.push(y);
        }
    }
}
int main(){
    cin >> n >> m;
    for (int i = 0; i < m; i ++ ) {
        int x, y; cin >> x >> y;
        insert(x, y);
    }
    topsort();
    int ans = 0;
    for (int i = 1; i <= n;  i ++ )
        if (otd[i] == 0)
            ans = (ans + f[i]) % MOD;
    cout << ans;
}

```
## 洛谷：P2853 [USACO06DEC] Cow Picnic S: 集中所有奶牛到牧场
这题我们从两个方向去思考，然后分别考虑它们的时间复杂度，然后我们`根据时间复杂度`来对`这一类`题目的`总结`

1. 从牧场出发进行 DFS 的角度：我们通过反向建边，然后遍历每一个牧场，去记录它们是否能够到达每一头奶牛，即 cnt[x] = k 时，该牧场能够作为一个符合题意的牧场
2. 从奶牛的角度出发进行 DFS：我们对每个奶牛进行 DFS，定义 f[x] 为 x 牧场`总共能够被 f[x] 奶牛遍历到`，那么 f[x] = k，代表所有的奶牛都能遍历到该牧场，那么该牧场为符合题意的牧场

那么角度 1 的时间复杂度为 O(n^2)，角度 2 的时间复杂度为 O(n * k) 所以我们应该去`选择角度 2`

总结：
1. 当我们觉得一个东西时间复杂度过高，或者正向遍历的做法是错误的时候，我们可以反向思考。看看能不能减少复杂度
2. 当我们不知道如何下手时，我们多增加几个数据结构，比如 f[x] 代表什么含义，能够符合题意并帮助解题，这样的话可能使我们想到解题方法。

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
int k, n, m;
int f[N], v[N];
void insert(int x, int y){
    G[x].push_back({y});
}
void DFS(int x) {
    v[x] = 1, f[x] ++ ;
    for (auto [y] : G[x]) {
        if (v[y]) continue;
        DFS(y);
    }
}
int main(){
    memset(f, 0, sizeof f);
    vector<int> nums;
    cin >> k >> n >> m;
    for (int i = 0; i < k; i ++ ) {
        int x; cin >> x;
        nums.push_back(x);
    }
    for (int i = 0; i < m; i ++ ) {
        int x, y; cin >> x >> y;
        insert(x, y);
    }
    for (auto x : nums) {
        memset(v, 0, sizeof v);
        DFS(x);
    }
    int ans = 0;
    for (int i = 1; i <= n; i ++ )
        if (f[i] == k) ans ++;
    cout << ans;
}
```


## 洛谷：P3385 【模板】负环
关于此题的`关键理解`：

这题的关键就是，
1. 如果你是去判断从 1 出发是否有负环，则你一定要先进行初始化： `memset(d, 0x3f, sizeof d), d[1] = 0` 
2. 而如果你要判断全图是否有负环，则你需要：`memset(d, 0, sizeof d)` 

你对哪几个点`一起`进行判负环，你就要初始化哪几个点的 d[x] = 0，否则你将更新不了它的邻边。举例：

`1 到 2 的边权为 3`，`2 到 1 的边权为 -6`，那么是存在负环的，因为`一开始只有 1 号节点在队中`然而你不初始化 d[2] = INF 的话，那么从 1 开始进行搜索的话就无法对 2 进行更新，因为 `不满足 d[1] + 3 < d[2]`。所以一`定要先进行初始化`。

而如果你是判断整个图是否有负环，那么你只需进行初始化`memset(d, 0, sizeof d)`，因为所有的节点都应该入队，那么每次`肯定会先拿出边权为负值`的边进行更新。

那么我们做个`通用的一般性总结`：有`哪些节点入队`，那么哪些 d[x] 就应该初始化为 0，而`其他的节点`的 d[other] 就应该初始化为 INF。

这题还有一个容易忽略的`特殊条件`：在读入边时，如果`边权大于等于零`，则该边为一个`双向边`，`否则为单向边`.

此题代码如下
```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
struct Edge{
    int y, z;
    Edge() {}
    Edge(int y, int z) : y(y), z(z) {}
};
vector<Edge> G[N];
int inq[N], nums[N], d[N];
int T, n, m;
void insert(int x, int y, int z) {
    G[x].push_back({y, z});
}
bool SPFA() {
    queue<int> q;
    memset(d, 0x3f, sizeof d);
    memset(inq, 0, sizeof inq);
    memset(nums, 0, sizeof nums);
    q.push(1), d[1] = 0, inq[1] = 1, nums[1] ++ ;
    while (q.size()) {
        int x = q.front(); q.pop(), inq[x] = 0;
        for (auto [y, z] : G[x]) {
            if (d[y] > d[x] + z) {
                d[y] = d[x] + z;
                if (!inq[y]) q.push(y), inq[y] = 1, nums[y] ++ ;
                if (nums[y] >= n) return 0;
            }
        }
    }
    return 1;
}
int main() {
    cin >> T;
    while (T -- ) {
        cin >> n >> m;
        for (int i = 1; i <= n; i ++ ) G[i].clear();
        for (int i = 0; i < m; i ++ ) {
            int x, y, z; cin >> x >> y >> z;
            if (z >= 0) insert(x, y, z), insert(y, x, z);   // 这是题目特别规定的 双向边
            else insert(x, y, z);       // 单向边
        }
        if (SPFA()) cout << "NO";
        else cout << "YES";
        cout << endl;
    }
    return 0;
}
```
## 437. 路径总和 III 
先学会暴力写法再去写优化写法，这种题目`连暴力写法都不会，还写个屁的优化`。

对该二叉树的每个节点都进行一次进行 DFS，进行计算`从当前节点`出发的`路径中` 有多少路径和为 tar。

所以对于子路径或者子数组这种连续的东西来说，我们一定要先固定它的起点或者固定它的结尾来统计以它为起点(或终点) 的所有的路径的性质
```c++
LL ans = 0, tar;
void DFS(TreeNode *p, LL sum) {
    LL cur = sum + p->val;
    if (cur == tar) ans ++ ;
    if (p->left) DFS(p->left, cur);
    if (p->right) DFS(p->right, cur);
    return;
}
vector<TreeNode*> arr;
void fun(TreeNode *p) {
    arr.push_back(p);
    if (p->left) fun(p->left);
    if (p->right) fun(p->right);
}
int pathSum(TreeNode* root, int targetSum) {
    if (root == NULL) return 0;
    tar = targetSum;
    fun(root);
    for (auto e : arr) {
        DFS(e, 0);
    }
    return ans;
}
```

`动态规划 + 前缀和优化`：我们计算以当前节点 p 为末尾节点的符合题意的路径有多少条。

当前节点使用 map 记录的前缀和时`有限制条件`：这里由于是二叉树，不是数组，所以我们在在 map 时，只能去使用`从根节点到当前节点为止`的路径上的前缀和，而`不能使用其他分叉出去`的`路径`的前缀和，所以我们`递归完当前的子树`后需要将从根到当前节点的前缀和`删除`。

那么我们可以写出以下代码：
```c++
unordered_map<LL, LL> mp;
LL tar, res = 0;
void DFS(TreeNode *p, LL sum) {
    if (p == NULL) return;
    sum += p->val;
    res += mp[sum - tar];       // 前缀和 sum - tar 的数量为 mp[sum - tar] 那么以当前节点为结尾的和为 tar 的路径共有 mp[sum - tar] 条，加入答案即可。
    mp[sum] ++ ;        // 该路径上的前缀和为 sum 的节点数量 + 1
    DFS(p->left, sum);
    DFS(p->right, sum);
    mp[sum] -- ;        // `递归完当前的子树` 后需要将从根到当前节点的前缀和 `删除` 。
}
int pathSum(TreeNode* root, int targetSum) {
    mp[0] = 1, tar = targetSum;
    DFS(root, 0);
    return res;
}
```

## 洛谷：P2440 木材加工

经典的二分答案问题。因为判断是否符合题目要求的代码非常容易写，所以为二分答案。`如果判断答案是否符合题意非常难，则大概率不是二分法`。

直接上代码：
```c++
#include <bits/stdc++.h>
using namespace std;
int n, k;
vector<int> arr;
bool judge(int x) {     // 我们很容易可以判断个木材长度是否符合标准
    int cnt = 0;
    for (auto e : arr) 
        cnt += e / x;
    return cnt >= k;
}
int main(){
    cin >> n >> k;
    int l = 0, r = 0;
    for (int i = 0; i < n; i ++ ) {
        int x; cin >> x; r = max(r, x);
        arr.push_back(x);
    }
    while (l < r) {
        int mid = l + r + 1>> 1;    // 标准的二分模板
        if (judge(mid)) l = mid;
        else r = mid - 1;
    }
    cout << l;
    return 0;
}
```

## 2575. 找出字符串的可整除数组
这题是`数学 + 模运算`题目，需要掌握如下数学知识才能做出：

如果直接做这题，一定会爆 int，long long 的数据范围，所以我们要用模运算来解决：

一个整数可表示为 a × 10 + b 。
而又有：(a × 10 + b) % m = ((a × 10) % m + b) % m

所以我们可以用不断求模来避免`爆数据范围`

代码如下：
```c++
typedef long long LL;
vector<int> divisibilityArray(string word, int m) {
    int n = word.size();
    vector<int> ans(n);
    LL t = 0;
    for (int i = 0; i < n; i ++ ) {
        t = (t * 10 % m + word[i] - '0') % m;
        ans[i] = t == 0;
    }
    return ans;
}
```
## AcWing 3481. 阶乘的和
这题可以用爆搜，也可以用背包，我们这里介绍一下背包的写法，以后看到选或不选，一定要想到有背包的写法！！

`易错点`：
1. 注意 0 的阶乘 = 1，也要包括在阶乘中！
2. 最后一行是负数，而不是 -1 
3. f[0] 虽然为 1，但是 0 不能被表示出来，需要特判。我们初始化 f[0] = 1 是为了能够让其他状态能够正确的转移，而不是因为 0 符合题目要求。

代码需要简洁化的地方：
1. 求出背包数组 v[]：阶乘的计算

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10;
int f[N], v[N], cnt = 0;
int main(){
    int n;
    for (int i = 0, j = 1; j <= N; j *= ++ i)   // 这是求阶乘的简洁方法，需要记住
        v[cnt ++ ] = j;
    
    memset(f, 0, sizeof f);
    f[0] = 1;
    for (int i = 0; i < cnt; i ++ )     // 利用动态规划预处理
        for (int j = N; j >= v[i]; j -- )
            f[j] = f[j] || f[j - v[i]];
    while (cin >> n && n >= 0) {
        if (f[n] && n != 0) cout << "YES";      // 我们需要对 0 进行特判，我们初始化 f[0] = 1 是为了能够让其他状态能够正确的转移，而不是因为 0 符合题目要求。
        else cout << "NO";
        cout << endl;
    }
    
    return 0;
        
}
```

DFS 搜索做法：

`特别注意的易错点`：如果题目的输入类型是`多组`数据的情况下，那么变量可能会`被上一组数据所污染`，当前的答案就会出错，所以我们一定要`重新设置变量`。

代码：

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6;
int flag = 0, nums[N], cnt = 0;
int tar;
void DFS(int index, int sum) {
    if (index < 0) return;
    if (sum == tar) { flag = 1; return ; }
    if (flag || sum > tar) return;   // 答案剪枝
    
    DFS(index - 1, sum + nums[index]);
    
    DFS(index - 1, sum);
}

int main(){
    for (int i = 0, j = 1; j <= N; j *= ++ i) 
        nums[cnt ++ ] = j;
    while (cin >> tar && tar >= 0) {
        flag = 0;       // flag 必须每组数据都被重新设置，否则就会被污染
        DFS(cnt - 1, 0);
        if (flag && tar != 0) cout << "YES";
        else cout << "NO";
        cout << endl;
    }
}
```

## ACwing 3480. 棋盘游戏
这是一个难题：`动态规划 + 最短路SPFA 或 Dij`

思想：我们不能简单的记录 `d[x][y]` 为 (x, y) 距离 (sx, sy) 的距离，而是要多加上一个状态： d[x][y][state] 代表 (x, y) 处于 state 的状态下，距离 (sx, sy) 的距离。因为`由于状态的不同`，可能会导致有些最短路是可以往回走，即`沿着重复路径进行重复走`的时候才是`最小的花费`代价。

我们用 Dij 来做一下试试：

用 Dij 的话要特别注意`小根堆`的`语法格式`：小根堆则重载 > 号，大根堆则需要重载 < 号

或者利用 cmp 结构体：

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
struct Node{
    int dis, x, y, state;
    Node () {}
    Node (int dis, int x, int y, int state) : dis(dis), x(x), y(y), state(state) {}     
    bool operator > (const Node &other) const {     // 小根堆则重载 > 号
        return dis > other.dis;
    }
    bool operator < (const Node &other) const {     // 大根堆则重载 < 号
        return dis < other.dis;
    }
};
int G[6][6];
int v[6][6][5];         // 由题目的 state 计算公式可得， state 只能在区间 [1, 4] 内
int d[6][6][5]; 
int dirs[4][2] = { {0, 1}, {0, -1}, {1, 0}, {-1, 0} };
int sx, sy, tx, ty, ans;
int main() {
    memset(v, 0, sizeof v);
    memset(d, 0x3f, sizeof d);
    for (int i = 0; i < 6; i ++ )
        for (int j = 0; j < 6; j ++ )
            cin >> G[i][j];
    cin >> sx >> sy >> tx >> ty;
    priority_queue<Node, vector<Node>, greater<Node> > q;
    q.push({0, sx, sy, 1}), d[sx][sy][1] = 0;
    while (q.size()) {
        auto [dis, x, y, state] = q.top(); q.pop();
        if (v[x][y][state]) continue; v[x][y][state] = 1;
        for (int i = 0; i < 4; i ++ ) {
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex >= 6 || nex < 0 || ney >= 6 || ney < 0) continue;
            int cost = state * G[nex][ney], neState = cost % 4 + 1;
            if (d[nex][ney][neState] > dis + cost) {
                d[nex][ney][neState] = dis + cost;
                q.push({d[nex][ney][neState], nex, ney, neState});
            }
        }
    }
    ans = INF;
    for (int i = 1; i < 5; i ++ )       // 由题目的 state 计算公式可得， state 只能在区间 [1, 4] 内
        ans = min(ans, d[tx][ty][i]);
    cout << ans;
    return 0;
}
```

为了避免重载运算符导致的错误，我们利用 SPFA 算法来做这题

```c++
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
struct Node{
    int x, y, state;
    Node() {}
    Node(int x, int y, int state) : x(x), y(y), state(state) {}
};
const int dirs[4][2] = { {0, 1}, {0, -1}, {1, 0}, {-1, 0} };
int d[6][6][5], inq[6][6][5];
int G[6][6];
int sx, sy, tx, ty;
int main(){
    memset(d, 0x3f, sizeof d);
    memset(inq, 0, sizeof inq);
    for (int i = 0; i < 6; i ++ )
        for (int j = 0; j < 6; j ++ )
            cin >> G[i][j];
    cin >> sx >> sy >> tx >> ty;
    queue<Node> q;
    q.push({sx, sy, 1}); inq[sx][sy][1] = 1, d[sx][sy][1] = 0;
    while (q.size()) {
        auto [x, y, state] = q.front(); q.pop(), inq[x][y][state] = 0;
        for (int i = 0; i < 4; i ++ ) {
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex < 0 || nex >= 6 || ney < 0 || ney >= 6) continue;
            int cost = G[nex][ney] * state;
            int neState = cost % 4 + 1;
            if (d[nex][ney][neState] > d[x][y][state] + cost) {
                d[nex][ney][neState] = d[x][y][state] + cost;
                if (!inq[nex][ney][neState]) q.push({nex, ney, neState}), inq[nex][ney][neState] = 1;
            }
        }
    }
    int ans = INF;
    for (int i = 1; i < 5; i ++ )
        ans = min(ans, d[tx][ty][i]);
    cout << ans;
    return 0;
}
```

计算模拟题：

## 592. 分数加减运算
这题告诉我们：多做模拟题，练习输入输出的处理，如何处理分数的运算以及最后的化简
1. 输入输出处理：利用 while() 循环 + 单独处理符号位(用一个 int 类型数据进行存储)
2. 分数运算与化简：我们`允许分数运算多次，但是只允许化简一次`。由于允许运算多次，那么分子分母势必会非常大，所以一定要用 long long 来存储分子与分母，防止溢出，而最后只需要用 gcd 处理一次即可。`注`：在进行计算时，一定不要用`最小公倍数进行通分`计算，而是直接让两个分母相乘即可。

细节：我们一定要
直接在代码中体会模拟：
```c++
typedef long long LL;
string fractionAddition(string s) {
    LL x = 0, y = 1;        // 初始化分数为 `0/1`
    int i = 0, n = s.size();
    while (i < n) {
        LL x1 = 0, sign = 1;        // 单独用 sign 来处理 正负号，只有分子带正负号，如果 `不带符号位则默认为正号`，初始化为 1
        if (s[i] == '-' || s[i] == '+')
            sign = s[i] == '-' ? -1 : 1, i ++ ;     
        while (i < n && isdigit(s[i]))
            x1 = x1 * 10 + s[i] - '0', i ++ ;
        x1 = sign * x1;
        i ++ ;      // 跳过除号： `/`
        long long y1 = 0;
        while (i < n && isdigit(s[i]))
            y1 = y1 * 10 + s[i] - '0', i ++ ;
        x = x * y1 + x1 * y;
        y *= y1;
    }
    LL g = gcd(abs(x), y);      // 0 与任意的 x 的 gcd 都是 x，特别注意我们这里
    return to_string(x / g) + "/" + to_string(y / g);
}
```

## PAT B1034 有理数四则运算
同上面的分数加减运算类似，此题也是一个很好的模拟题
网址 : https://pintia.cn/problem-sets/994805260223102976/exam/problems/994805287624491008?type=7&page=0

## 640. 求解方程

先上代码`再总结`：
```c++
string solveEquation(string s) {
        int a = 0, b = 0;       // a 存储 x 的系数，b 存储常数，以此化为一元方程的一般式：a * x + b = 0
        int i = 0, n = s.size(), sign = 1;      // 默认符号位，如果 `不带符号位则初始化为 默认符号位`。记 `= 号` 左侧的符号位为
        while (i < n) {
            if (s[i] == '=') {      // 当前位为分隔符 = 号，直接翻转默认符号位
                sign = -sign, i ++ ;
                continue;
            }
            int curS = sign, num = 0;   // 将 curS 先初始化为默认符号位
            bool isNum = 0;     // 判断 x 之前有无系数
            if (s[i] == '-' || s[i] == '+')         // 如果有符号位则对 curS 进行赋值
                curS = s[i] == '-' ? -sign : sign, i ++ ;

            while (i < n && isdigit(s[i]))  // 生成当前的常数 或者 是 x 的系数
                num = 10 * num + s[i] - '0', i ++ , isNum = 1;
            
            if (i < n && s[i] == 'x')           // 如果当前为 x
                a += isNum ? curS * num : curS, i ++ ;    // 判断 x 之前有无系数，即 isNum
            else        // 
                b += curS * num;
        }
        if (a == 0) return b == 0 ? "Infinite solutions" : "No solution";
        return "x=" + to_string(-b / a);
    }
```
总结：我们就是需要按照一个固定的思路一步一步地进行 while() 循环，即循环体内一定是`按照一种特定的步骤`进行生成答案的

步骤如下：
1. 判断 s[i] 是否为 =，是则翻转默认的 sign 符号
2. 初始化当前符号 curS，再判断有无符号位
3. 判断是否有数字 isNum
4. 判断是否为 x 前的系数 ai，还是常数 bi。同时进行累加 a = Σai 或 b = Σbi.


## 二次方程计算器
网址：https://www.nowcoder.com/practice/071f1acaada4477f94193f8c0b9054f4?tpId=62&tqId=29449&tPage=1&rp=1&ru=/ta/sju-kaoyan

方法和 `640. 求解方程` 差不多，关键的技巧就在于：我们判断 i 后面的字符的字符时，我们不用去详细的考虑到底当前的状态是什么，我们只需要牢记一个点：当你想访问 s[i + 2] 时，你一定要`先判断 i + 2 < n`，这是必须的，也是我们这种 `往前预判断` 的代码比用技巧。即在`往前探索时`，必须先去`判断是否越界`，这一个`技巧很重要`，可以`简化代码与思路`

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
string s;
int main() {
    cin >> s;
    int n = s.size(), sign = 1, i = 0;
    int a = 0, b = 0, c = 0;
    while (i < n) {
        if (s[i] == '=') {
            sign = -sign, i ++ ;
            continue;
        }

        int curS = sign, isNum = 0, num = 0;
        if (s[i] == '-' || s[i] == '+')
            curS = s[i] == '-' ? -sign : sign, i ++ ;

        while (i < n && isdigit(s[i]))      // 获取数字
            num = num * 10 + s[i] - '0', i ++, isNum = 1;

        if (i < n && s[i] == 'x' ) {        // 判断为二次项或一次项
            if (i + 1 < n && s[i + 1] == '^')       // 向前探步，看是否是 二次项
                a += isNum ? curS * num : curS, i += 3; 
            else                                    // 对一次项的
                b += isNum ? curS * num : curS, i ++ ;  
        }
        else                                // 判断为常数项
            c += curS * num;
    }
    if (b * b - 4 * a * c < 0) cout << "No Solution";       // 计算 答案 
    else {
        double deta = b * b - 4 * a * c;
        double x1 = (-b - sqrt(deta)) / (2 * a), x2 = (-b + sqrt(deta)) / (2 * a);
        printf("%.2lf %.2lf", x1, x2);
    }
    return 0;
}
```



## acwing : 3482. 大数运算

背住板子即可
```c++
#include <bits/stdc++.h>
using namespace std;

int cmp(vector<int> &a, vector<int> &b) {   // 比较的是两者的绝对值大小
    if (a.size() > b.size()) return 1;
    if (a.size() < b.size()) return -1;
    
    for (int i = a.size() - 1; i >= 0; i -- ) {     // cmp 与 print 中特别的地方就是要从 size - 1 处开始进行判断
        if (a[i] > b[i]) return 1;
        else if (a[i] < b[i]) return -1;
    }
    return 0;
}
vector<int> add(vector<int> &a, vector<int> &b) {
    vector<int> c;
    for (int i = 0, t = 0; i < a.size() || i < b.size() || t; i ++ ) {      // add 中特别的地方就是 `t 应该处于 for 内的判断条件` 以处理最后的进位超过 a 和 b 的最大 size 
        if (i < a.size()) t += a[i];
        if (i < b.size()) t += b[i];
        c.push_back(t % 10);
        t /= 10;
    }
    return c;
}
vector<int> sub(vector<int> &a, vector<int> &b) {
    vector<int> c;
    for (int i = 0, t = 0; i < a.size(); i ++ ) {
        if (i < a.size()) t += a[i];
        if (i < b.size()) t -= b[i];
        c.push_back((t + 10) % 10);
        t = t < 0 ? -1 : 0;
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();     // sub 中特别的地方就是在最后应该要去处理前导 0 。注意这里是 c.size() > 1，而没有等于符号，因为如果等于 1 再 pop_back() 就把最后一个 0 都搞没了
    return c;
}
vector<int> mul(vector<int> &a, vector<int> &b) {
    vector<int> c(a.size() + b.size());
    for (int i = 0; i < a.size(); i ++ )        // 通过双重循环进行存储结果
        for (int j = 0; j < b.size(); j ++ )    
            c[i + j] += a[i] * b[j];
    for (int i = 0, t = 0; i < c.size(); i ++ ) {   // 将运算结果进行化简
        t += c[i];
        c[i] = t % 10;
        t /= 10;
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}
vector<int> div(vector<int> &a, int b, int &r) {
    vector<int> c; r = 0;
    for (int i = A.size() - 1; i >= 0; i -- ) {     // 从最高位开始 div
        r = r * 10 + a[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
void print(vector<int> a) {
    for (int i = a.size() - 1; i >= 0; i -- )       // 和 cmp 中一样，要从 size() - 1 处开始 print
        cout << a[i];
    cout << endl;
}
int main(){
    string s1, s2; cin >> s1 >> s2;
    int sa = 1, sb = 1;     // 将 数值部分 和 符号部分 分开存储
    if (s1[0] == '-') sa = -1, s1 = s1.substr(1);
    if (s2[0] == '-') sb = -1, s2 = s2.substr(1);
    
    vector<int> a, b;
    for (int i = s1.size() - 1; i >= 0; i -- ) a.push_back(s1[i] - '0');
    for (int i = s2.size() - 1; i >= 0; i -- ) b.push_back(s2[i] - '0');
    
    if (sa * sb > 0) {
        if (sa < 0) cout << '-';
        print(add(a, b));
        
        bool flag = 0; 
        if (cmp(a, b) < 0) 
            flag = 1, swap(sa, sb), swap(a, b);     // 交换，
        // cout << sa << endl;
        if (sa > 0 && flag || sa < 0 && !flag && a != b) cout << '-';
        print(sub(a, b));
        
        print(mul(a, b));
    }
    else {
        bool flag = 0;
        if (cmp(a, b) < 0) 
            flag = 1, swap(sa, sb), swap(a, b);
        if (sa < 0 && a != b) cout << "-";
        print(sub(a, b));
        
        if (sa > 0 && flag || sa < 0 && !flag) cout << '-';
        print(add(a, b));
        
        if (s1 != "0" && s2 != "0") cout << '-';
        print(mul(a, b));
    }
    return 0;
}
```

## LeetCode 43. 字符串相乘
直接用模板：
```c++
vector<int> mul(vector<int> &a, vector<int> &b) {       // 标准的 mul 模板
    vector<int> c(a.size() + b.size());
    for (int i = 0; i < a.size(); i ++ ) 
        for (int j = 0; j < b.size(); j ++ )
            c[i + j] += a[i] * b[j];
    for (int i = 0, t = 0; i < c.size(); i ++ ) {
        t += c[i];
        c[i] = t % 10;
        t /= 10;
    }
    while (c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}
string multiply(string s1, string s2) {
    vector<int> a, b, c;
    for (int i = s1.size() - 1; i >= 0; i -- ) a.push_back(s1[i] - '0');
    for (int i = s2.size() - 1; i >= 0; i -- ) b.push_back(s2[i] - '0');
    
    c = mul(a, b);
    string s(c.size(), '0');
    for (int i = s.size() - 1, j = 0; i >= 0; i -- , j ++ )
        s[i] = c[j] + '0';
    return s;
}
```

## LeetCode 224. 基本计算器 227. 基本计算器 II
这个东西在刷题笔记中去寻找做法

总结1 ：一个累计字符串中数字的做法：

两种方法：
1. 利用探步法：
```c++
int res = 0, isNum = 0;
while (i < n && isdigit(s[i]))
    res = res * 10 + s[i] - '0', i ++ , isNum = 1;
if (isNum)
// process
```

2. 利用库函数法：
```c++
int res = 0, isNum = 0, j = i;
while (j < n && isdigit(s[i])) j ++ , isNum = 1;
string t = s.substr(i, j - i); stringstream(t) >> res;      // 利用库函数进行赋值
i = j;
```

总结二：这种模拟问题一般是`数值位和符号位`分开考虑！！！
```c++
int sign = 1;
int curS = sign;
if (s[i] == '+' || s)
```


## 1360. 日期之间隔几天

```c++
#include<bits/stdc++.h>
using namespace std;
int a[13]={0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};//把每月的天数求出来
bool isLeap(int y) {        // 闰年的二月为 29 天
    if (y % 4 == 0 && y % 100 !=0 || y % 400 == 0) return 1;
    return 0;
}

int day(int n)
{
    int y = n / 10000;          // 年
    int m = n % 10000 / 100;    // 月
    int d = n % 100;            // 日

    a[2] = isLeap(y) ? 29 : 28;     // 判断，并赋值给 a[2] 
    
    while ( -- m) d += a[m];       // 把每个月的天数相加
    while ( -- y) d += isLeap(y) ? 366 : 365;       // 闰年366天，平年365天
    
    return d;       // 返回一共的天数
}
int main()
{
    int a, b;
    while(cin >> a >> b)
        cout << abs(day(a) - day(b)) + 1 <<endl;    //由题所示 "如果两个日期是连续的我们规定他们之间的天数为两天" ，因此我们加一
}
```

## 1360. 日期之间隔几天
这里要积累一个模拟`常用`的跳过无用字符的两个方法：`法二更为常用！！！`
```c++
// 法1：跳过无用字符
for (int i = 0; i < n; i ++ ) {
    if (!isdigit(s[i]) && !isalpha(s[i])) continue;     // 如果不是
}

// 法2：只积累有用字符：这个更好
for (int i = 0; i < n; i ++ ) {
    if (isdigit(s[i])) {
        // process
    }
    else if (isalpha(s[i])) {   // 可以用 isupper, islower
        // process
    }
}
```
```c++
int a[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};        // 由于每次都是从上一月开始
bool isLeap(int year) {
    return ((year % 400 == 0) || (year % 100 != 0 && year % 4 == 0));
}
int day(int date) {
    int y = date / 10000;
    int m = date / 100 % 100;
    int d = date % 100;

    a[2] = isLeap(y) ? 29 : 28;

    while ( -- m) d += a[m];        // 这里是先对 m 进行减处理，也就是从上一月开始累加，而 `本月` 的天数并不算入在内
    while ( -- y) d += isLeap(y) ? 366 : 365;   // 同理这里从上一年开始累加，而不将 `本年` 的天数算入
    return d;
}

int daysBetweenDates(string s1, string s2) {
    int date1 = 0, date2 = 0;
    for (int i = 0; i < s1.size(); i ++ ) {
        if (!isdigit(s1[i])) continue;
        date1 = date1 * 10 + s1[i] - '0';
    } 
    for (int i = 0; i < s1.size(); i ++ ) {
        if (!isdigit(s2[i])) continue;
        date2 = date2 * 10 + s2[i] - '0';
    }
    return abs(day(date1) - day(date2));
}
```

## 54. 螺旋矩阵
我们在这种题目中总结一个经验，即我们用 nex, ney `作为探步的辅助变量`，而`不是`直接将 x, y 变为 nex 和 ney：
1. 如果 nex, ney 符合探步要求，则不用进行方向的变换，然后改变 x, y 但是不用 nex 和 ney 来进行改变，而是直接利用原来的 dirs[][] 来进行改变，这就很具有代码的简洁性，因为我们只需让 nex 和 ney `作为判断的辅助变量`，而`不让`他成为 `赋值`的辅助变量。这就和不符合探步要求的代码能够同步。
2. 如果 不符合探步要求，则进行下一次正确的变换，然后改变 x, y

细节：此题和深度优先搜索不同，此题`最多`变换一个方向，如果变换完一个方向后仍不符合题意则说明结束了。

我们有两种方法判断结束：任选一种即可
1. 做完正确的变换后还是不符合题意则退出
2. 预先计算出总数，如果所有的数都被遍历过一次后则推出

代码如下：
```c++
int v[110][110];
const int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
vector<int> spiralOrder(vector<vector<int>>& mat) {
    memset(v, 0, sizeof v);
    int m = mat.size(), n = mat[0].size();
    vector<int> res;
    int i = 0, total = n * m, x = 0, y = 0;
    while (total -- ) {     
        v[x][y] = 1, res.push_back(mat[x][y]);
        int nex = x + dirs[i][0], ney = y + dirs[i][1];
        if (nex < 0 || nex >= m || ney < 0 || ney >= n || v[nex][ney])
            i = (i + 1) % 4;
        x = x + dirs[i][0], y = y + dirs[i][1];     // 我们不让 nex 进行直接的赋值
        // 如果使用方法 1 来判断结束，则可以利用 if (x < 0 || x >= m || y < 0 || y >= n || v[x][y]) break;
    }
    return res;
}
```

## 67. 二进制求和
在这个题目中有趣的是，我们可以用 reverse + for (auto) 这两个语法来分别将 s 倒序，和简便地遍历赋值 vector 而无需考虑它们的正反和大小关系

代码略


## 2834. 找出美丽数组的最小和
方法一：`贪心 + 暴力枚举`，这会超内存。

贪心思想：我们尽可能的选取较小的数字，从 1 开始枚举，如果符合题目要求则将当前数字 cur 加入答案中

注意枚举代码的写法：如果`暴力做法都不会做还做个屁的优化`
```c++
const int MOD = 1e9 + 7;
unordered_set<int> S;
int minimumPossibleSum(int n, int tar) {
    int ans = 0;
    for (int cur = 1, cnt = 0; cnt < n; ) {
        if (S.count(tar - cur)) { cur ++ ; continue; }
        S.insert(cur), ans = (ans + cur) % MOD, cur ++ , cnt ++ ;
    }
    return ans;
}
```

方法二： `贪心 + 数学公式`

贪心思想不变，我们用等差数列和公式来进行 O(1) 求解。代码解释如下

1. 在 tar 之前选取数字，则 `最多` 只能选 min(n, tar / 2) 个数字
2. 如果选够了(所选数量大于等于 n)，即 left_num >= n, 则返回 left_sum
3. 如果没选够，从 tar 开始去选，逐渐选够为止
```c++
int minimumPossibleSum(int n, int tar) {
    LL left_num = min(n, tar / 2);      // 在 tar 之前选取数字 `最多` 只能选 min(n, tar / 2) 个数字
    LL left_sum = (1 + left_num) * left_num / 2 % MOD;
    LL right_num = n - left_num;
    if (right_num <= 0) return left_sum;
    LL right_sum = (tar + tar + right_num - 1) % MOD * right_num / 2 % MOD;
    return (left_sum + right_sum) % MOD;
}
```

## 289. 生命游戏
这题的`关键`就是如同 BellMan Ford 算法一样，`必须`用一个 `backup` 矩阵，`不能`在`原`矩阵上进行`更新`，否则将会和题目的限制条件：同时发生死亡与复活相矛盾，即不能用`更新后`的状态来`更新当前`的状态。

代码略。

## 495. 提莫攻击
初看不知道，`一看吓一跳`，这是一个非常好的区间合并题目，我们直接用区间合并的模板，上代码：

```c++
int findPoisonedDuration(vector<int>& nums, int t) {
    int curL = nums[0], curR = nums[0] + t - 1;
    int ans = 0;
    for (int i = 1; i < nums.size(); i ++ ) {
        if (curR >= nums[i]) curR = nums[i] + t - 1;
        else ans += curR - curL + 1, curL = nums[i], curR = nums[i] + t - 1;
    }
    ans += curR - curL + 1;
    return ans;
}
```

## 498. 对角线遍历
该题的转换模式可以用一个字来形容：`优先`选什么，`次选`什么，这个就跟我们前面的解方程题目很像：我们`最`优先判断 `=`； `次`优先判断 `+，-` 号，然后`再次` 就是判断 `isNum` 最次才判断 是否是 `x` 的系数
    
这题就是:
1. 如果当前遍历`方向为右上`时：如果出界`变换方向`为`左下`后，我们的 x 和 y：`优先`让 y ++ . `如果 y 不能 ++ `，`才`让 x ++ 。
2. 同理，当前遍历`方向为左下`时：如果出界`变换方向`为`右上`后，我们优先让 x ++ , 如果 x 不可 ++ 才让 y ++ .

代码如下：
```c++
int dirs[2][2] = { {-1, 1}, {1, -1} };
vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
    int x = 0, y = 0, dir = 0;
    int m = mat.size(), n = mat[0].size();
    vector<int> res;
    for (int i = 0; i < n + m - 1; i ++ ) {     // 总共的对角线数量
        while (1) {
            res.push_back(mat[x][y]);
            int nex = x + dirs[dir][0], ney = y + dirs[dir][1];
            if (nex < 0 || nex >= m || ney < 0 || ney >= n) break;
            x = nex, y = ney;
        }
        // 出界后：
        if (dir == 0)   // 当前遍历`方向为右上`时, `优先`让 y ++ . `如果 y 不能 ++ `，`才`让 x ++ 
            if (y + 1 < n) y ++ ;   
            else x ++ ;
        else        // 当 前遍历 `方向为左下` 时, 优先让 x ++ , 如果 x 不可 ++ 才让 y ++ .
            if (x + 1 < m) x ++ ;
            else y ++ ;
        dir = (dir + 1) % 2;
    }
    return res;
}
```

## 537. 复数乘法
这题我们可以用三种方法来解决：
1. 逐步`根据先后顺序`模拟法
```c++
typedef pair<int, int> PII;
PII fun(string s) {     // 经典的 `顺序生成` 做法
    PII res;
    int i = 0, sign1 = 1, sign2 = 1, num1 = 0, num2 = 0;
    while (i < s.size()) {
        if (s[i] == '-') sign1 = -1, i ++ ;

        while (i < s.size() && isdigit(s[i]))
            num1 = num1 * 10 + s[i] - '0', i ++ ;
        
        i ++ ;
        
        if (i < s.size() && s[i] == '-') sign2 = -1, i ++ ;
        while (i < s.size() && isdigit(s[i]))
            num2 = num2 * 10 + s[i] - '0', i ++ ;

        if (s[i] == 'i') break;
    }
    res.first = sign1 * num1, res.second = sign2 * num2;
    return res;
}
string complexNumberMultiply(string s1, string s2) {
    PII a = fun(s1), b = fun(s2);
    int c1 = a.first * b.first - a.second * b.second, c2 = a.second * b.first + a.first * b.second;
    string ans;
    ans = to_string(c1) + '+' + to_string(c2) + 'i';
    return ans;
}
```

2. 利用 sscanf(): 由于给定的字符分割的地方是固定的，所以直接使用格式化输入可以做
```c++
string complexNumberMultiply(string s1, string s2) {
    int a1, a2, b1, b2;
    sscanf(s1.c_str(), "%d+%di", &a1, &a2);
    sscanf(s2.c_str(), "%d+%di", &b1, &b2);
    int c1 = a1 * b1 - a2 * b2, c2 = a1 * b2 + a2 * b1;
    return to_string(c1) + "+" + to_string(c2) + "i";
}
```

3. 利用 stringstream 同上
```c++
string complexNumberMultiply(string s1, string s2) {
    int a1, a2, b1, b2;
    char c;
    stringstream(s1) >> a1 >> c >> a2 >> c;     // 一定要让 c 作为分割符
    stringstream(s2) >> b1 >> c >> b2 >> c;
    
    int c1 = a1 * b1 - a2 * b2, c2 = a2 * b1 + a1 * b2;
    return to_string(c1) + '+' + to_string(c2) + 'i';
}
```
## 566. 重塑矩阵
该题是一个数学问题，对于一个矩阵来说，他的`行数为 m`, `列数为 n`，`按行`遍历时，若 num 为第 x 个数(`x 从 0 开始计数`)，那么他`所在行`为：x / n, `所在列`为 x % n。

那么就直接写出代码：
```c++
// 方法 1 : 按照生成矩阵的行和列依次遍历生成
vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
    int m = mat.size(), n = mat[0].size();
    vector<vector<int> > res(r, vector<int>(c));
    if (r * c != m * n) return mat;
    for (int i = 0; i < r; i ++ )       // 方法 1 ：用 i，j 来进行计算当前的 num 为第 x = i * c + r 个
        for (int j = 0; j < c; j ++ ) {
            int x = (i * c + j) / n, y = (i * c + j) % n;
            res[i][j] = mat[x][y];
        }
    return res;
}

// 方法 2 ：按 nums 为第 x 个数进行遍历生成，这种方法 `更加清晰`
vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) {
    int m = nums.size();
    int n = nums[0].size();
    if (m * n != r * c) return nums;

    vector<vector<int>> ans(r, vector<int>(c));
    for (int x = 0; x < m * n; x ++ )     
        ans[x / c][x % c] = nums[x / n][x % n];
    
    return ans;
}
```

## 547. 省份数量
该题可以用并查集来做，也可以用图来做，我们这里主要讲解一下，如何判断并查集中`不同集合`的`数量`？


1. 我们只需要判断每个集合的标志节点即可：即 p[i] == i 则代表了唯一一个集合。
2. 我们处理在 find() 函数中用到了 p[x]， 其他任何地方都不要使用 p[x] ！！！你要么用 px = find(x), 要么用 p[px] ，但是`绝对别用 p[x]`
代码如下：
```c++
int p[210];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
void merge(int x, int y) {
    int px = find(x), py = find(y);     // 特别注意， x 所在的集合不是 p[x]，而是 find(x) !!!
    if (px == py) return;
    p[px] = py;
}
int findCircleNum(vector<vector<int>>& G) {
    int n = G.size();
    for (int i = 0; i < n; i ++ ) p[i] = i;
    for (int x = 0; x < n; x ++ )
        for (int y = 0; y < n; y ++ ) {
            if (G[x][y]) merge(x, y);
        }
    int ans = 0;
    for (int i = 0; i < n; i ++ )       // 计算并查集中
        if (i == p[i]) ans ++ ;
    return ans;
}
```

## 牛客网：KY108 Day of Week
https://www.nowcoder.com/practice/a3417270d1c0421587a60b93cdacbca0?tpId=62&tPage=1&rp=1&ru=%2Fta%2Fsju-kaoyan&difficulty=&judgeStatus=&tags=&title=&sourceUrl=&gioEnter=menu

这里我们系统介绍一下日期的计算方式以及特点：
1. 我们的公元从`第一年`开始：0001-01-01 为`星期一`。`而不是`从公元 0000 年开始的，`是`从 0001 年开始的
2. 我们累计年数和月数时，一定不能从本年或本月开始逐渐累加，而是从上一年与上一个月开始逐渐累加，因为`当前年与当前月`的`所有`天数不能算入在答案内。

所以代码如下：
```c++
#include <bits/stdc++.h>
using namespace std;
int a[13] = {0, 31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30,31 };
unordered_map<string, int> mp1;
unordered_map<int, string> mp2;
bool isLeap(int y) {
    if (y % 400 == 0 || y % 4 == 0 && y % 100 != 0) return 1;
    return 0;
}
int main(){
    mp1["January"] = 1, mp1["February"] = 2, mp1["March"] = 3, mp1["April"] = 4, mp1["May"] = 5, mp1["June"] = 6, mp1["July"] = 7, mp1["August"] = 8, mp1["September"] = 9, mp1["October"] = 10, mp1["November"] = 11, mp1["December"] = 12;
    mp2[7] = "Sunday", mp2[1] = "Monday", mp2[2] = "Tuesday", mp2[3] = "Wednesday", mp2[4] = "Thursday", mp2[5] = "Friday", mp2[6] = "Saturday";
    int y, m, d;
    string s;
    while (cin >> d >> s >> y) {
        int m = mp1[s];
        a[2] = isLeap(y) ? 29 : 28;
        while ( -- m) d += a[m];        // 一定是先 -- m ，才能进行累加
        while ( -- y) d += isLeap(y) ? 366 : 365;   // 一定是先进行减去当前的年，才能进行累加
        int wk = (d - 1) % 7 + 1;       // 公元从 `第一年` 开始：0001-01-01 为 `星期一` 
        cout << mp2[wk] << endl;
    }
    return 0;
}
```
## LeetCode 885. 螺旋矩阵 III
想做出这道题，必须掌握一个规律：
1. 进行的方向：一共只有 4 种不同的方向
2. 在每个方向上走的距离规律：1, 1, 2, 2, 3, 3, 4, 4, 5, 5, ... 。也就是说：向右走 1 步，向下走 1 步，向左走 2 步，向上走 2 步：此时在一次`旋转循环`中完整地走了一次，那么进行 `下一次` `旋转循环` ：向右走 3 步，向下走 3 步，向左走 4 步，向上走 4 步。如此就得到了一个规律：方向旋转的循环每 4 次进行一轮新的方向旋转循环。而步数则是循环中每走两步就加 1.

代码技巧：我们`允许`坐标`越界`，但我们`只`记录下`没有越界`的坐标。

关键：关键就是`每次只走一步`，而`不是``一次性`走多步

关键：在四个方向上循环时，后两个方向的步数比前两个方向的步数多一

由此我们写出代码：
```c++
vector<vector<int>> spiralMatrixIII(int m, int n, int x, int y) {
    vector<vector<int> > res;   
    res.push_back({x, y});          // 将原始的 (x, y) 坐标给加入
    for (int k = 1; k < 2 * (m + n); k += 2) {      // 考虑到如果我们的点是从边界（比如矩阵的右下角）开始的，那么最大步长不能超过 2 * (m + n)，然后我们每次都对步长 k 进行 1，3，5，7 这样对应于 旋转循环中 第一次向右 的 前进步数。
        int dirs[4][3] = { {0, 1, k}, {1, 0, k}, {0, -1, k + 1}, {-1, 0, k + 1} };  // 设置好每个方向和前进步数
        for (auto dir : dirs) {     // 进行旋转循环，直接按照套路出牌即可
            int dx = dir[0], dy = dir[1], dk = dir[2];
            for (int i = 0; i < dk; i ++ ) {    // 我们每次的前进步数为 1，一共前进 dk 次，`不能` `一次性` 直接前进 dk 步，因为要考虑到将沿途的坐标加入 res 中，并且还得判断坐标是否在 矩阵的 界限内。
                x += dx, y += dy;
                if (x < 0 || x >= m || y < 0 || y >= n) continue;
                res.push_back({x, y});
                if (res.size() == m * n) return res;
            }
        }
    }
    return res;
}
```

## LeetCode 946. 验证栈序列
这是一个模拟题，我先写的代码能通过，但是不好，还是得向官解学习：

代码`技巧`：我们每次都以入栈为先，出栈为后。我的代码`之所以不好`就是以出栈为先，入栈为后
```c++
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    vector<int> stk;
    int n = pushed.size();
    for (int i = 0, j = 0; i < n; i ++ ) {
        stk.push_back(pushed[i]);       // 先进行入栈，然后才出栈
        while (stk.size() && j < n && stk.back() == popped[j])  // 每次都要先判下标：stk.size() 和 j < n;
            stk.pop_back(), j ++ ;
    }
    return stk.empty();     // 如果最后栈可全部弹出，则说明是一个正确的序列。
}
```

## LeetCode 874. 模拟行走机器人
这题就是低级版的 `885. 螺旋矩阵 III`
直接上代码：

```c++
set<PII> S;
int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
int robotSim(vector<int>& com, vector<vector<int>>& ob) {
    for (auto e : ob) S.insert({e[0], e[1]});
    int dir = 0, x = 0, y = 0, ans = 0;
    for (auto e : com) {
        if (e == -1) dir = (dir + 1) % 4;
        else if (e == -2) dir = (dir + 3) % 4;
        else {
            for (int i = 0; i < e; i ++ ) {
                int nex = x + dirs[dir][0], ney = y + dirs[dir][1];
                if (S.count({nex, ney})) break;
                x = nex, y = ney;
                ans = max(ans, x * x + y * y);
            }
        }
    }
    return ans;
}
```

## 950. 按递增顺序显示卡牌
这题不能说是毫无思路，简直说是一点思路都没有！！！

做法：我们先假设我们`已经`得到了一个`答案数组`：id[] 这个数组存放的是`答案数组的下标`，而不是答案数组本身，我们先默认这就是答案。

那么我们模拟这个过程：

模拟的`关键`就是给 `当前的nums[i]` 绑定上对应的正确下标。

那么这个 nums[i] 对应的下标应该就是每次抽取的第一张卡片，然后下标数组要模拟队首出队，队尾入队的操作，这样就得到了下一个 nums[i + 1] 的对应的下标

```c++
vector<int> deckRevealedIncreasing(vector<int>& nums) {
    int n = nums.size();
    deque<int> id(n);
    vector<int> res(n);
    for (int i = 0; i < n; i ++ ) id[i] = i;
    sort(nums.begin(), nums.end());

    for (int i = 0; i < n; i ++ ) {
        res[id.front()] = nums[i];          // 给 nums[i] 绑定上正确的下标
        id.pop_front();
        if (id.size()) {
            id.push_back(id.front());
            id.pop_front();
        }
    }
    return res;
}
```
## 542. 01 矩阵 && 1162. 地图分析
多元最短路可以写，但是这题还是用 `动态规划方法更爽`
直接上动态规划的方法：

我们在四个方向上分别进行遍历来用  `f[i][j] = min(f[i][j], f[i - 1][j] + 1)` 这种四个方向上的类似方式来进行动态规划即可。

直接上其中一个的代码：1162. 地图分析

只要注意他的`遍历顺序`以及动态规划方程`边界`的一些细节即可
```c++
int maxDistance(vector<vector<int>>& mat) {
        int n = mat.size();
        vector<vector<int> > f(n, vector<int>(n, INF));
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ ) {
                if (mat[i][j] == 1) f[i][j] = 0;
                else {
                    if (i - 1 >= 0) f[i][j] = min(f[i][j], f[i - 1][j] + 1);
                    if (j - 1 >= 0) f[i][j] = min(f[i][j], f[i][j - 1] + 1);
                }
            }

        for (int i = 0; i < n; i ++ )
            for (int j = n - 1; j >= 0; j -- ) {
                if (mat[i][j] == 1) f[i][j] = 0;
                else {
                    if (i - 1 >= 0) f[i][j] = min(f[i][j], f[i - 1][j] + 1);
                    if (j + 1 < n) f[i][j] = min(f[i][j], f[i][j + 1] + 1);
                }
            }
        
        for (int i = n - 1; i >= 0; i -- )
            for (int j = 0; j < n; j ++ ) {
                if (mat[i][j] == 1) f[i][j] = 0;
                else {
                    if (i + 1 < n) f[i][j] = min(f[i][j], f[i + 1][j] + 1);
                    if (j - 1 >= 0) f[i][j] = min(f[i][j], f[i][j - 1] + 1);
                }
            }

        for (int i = n - 1; i >= 0; i -- )
            for (int j = n - 1; j >= 0; j -- ) {
                if (mat[i][j] == 1) f[i][j] = 0;
                else {
                    if (i + 1 < n) f[i][j] = min(f[i][j], f[i + 1][j] + 1);
                    if (j + 1 < n) f[i][j] = min(f[i][j], f[i][j + 1] + 1);
                }
            }

        int ans = 0;
        for (int i = 0; i < n; i ++ ) 
            for (int j = 0; j < n; j ++ ) {
                    ans = max(ans, f[i][j]);
            }

        return ans == INF || ans == 0 ? -1 : ans;
    }
```

## 802. 找到最终的安全状态
方法：反向建边 + 拓扑排序

我们将所有的边进行反向建边，然后让 `反向建边之后的图` 进行拓扑排序。

将最后能够拓扑排序节点进行保存即可：

代码如下：
```c++
const int N = 1e4 + 10;
class Solution {
public:
    struct Edge{
        int y;
        Edge() {}
        Edge(int y) : y(y) {}
    };
    int n;
    vector<Edge> G[N];
    vector<int> ans;
    int ind[N]; 
    void insert(int x, int y) {
        G[x].push_back(y); ind[y] ++ ;
    }
    void topsort() {
        queue<int> q;
        for (int i = 0; i < n; i ++ )
            if (ind[i] == 0) q.push(i);
        while (q.size()) {
            for (int i = q.size(); i; i -- ) {
                auto x = q.front(); q.pop();
                ans.push_back(x);
                for (auto [y] : G[x]) 
                    if ( -- ind[y] == 0) q.push(y);
            }
        } 
    }
    vector<int> eventualSafeNodes(vector<vector<int>>& tmp) {
        n = tmp.size(); memset(ind, 0, sizeof ind);
        for (int i = 0; i < n; i ++ ) 
            for (auto y : tmp[i]) 
                insert(y, i);
        topsort();
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```

## 841. 钥匙和房间
经典的 `DFS（或BFS） + 节点计数`

也可以用 `DFS + 联通分支` 来写，我这里利用联通分支来做的：

代码如下：
```c++
int v[N];
vector<vector<int>> G;
void DFS(int x) {
    if (v[x]) return; v[x] = 1;
    for (auto y : G[x]) {
        if (v[y]) continue;
        DFS(y);
    }
}
bool canVisitAllRooms(vector<vector<int>>& _G) {
    int n = _G.size(); G = _G;
    memset(v, 0, sizeof v);
    int cnt = 0, flag = 1;
    for (int i = 0; i < n; i ++ ) {     
        if (!v[i]) DFS(i), cnt ++ ;
    }
    return cnt == 1;
}
```


## 797. 所有可能的路径
这是回溯，不是深搜

该题只需要从 0 -> n - 1 的所有路径，所以可以 `不用 v[]`

我们要搞清楚为什么不用 `v[]` ，这题是回溯，而不是深搜，所以我们不能说可以不用 v[] 而是一定不需要用到 v[]，回溯算法`可以`需要去搜索`已经访问过`的节点，所以用 v[] 来记录访问过的节点根本无用。m
```c++
vector<vector<int>> G;
vector<vector<int> > res;
vector<int> t;
int n;
void DFS(int x) {
    if (x == n - 1) {
        res.push_back(t);
        return;
    }
    for (auto y : G[x]) {  
        t.push_back(y);        
        DFS(y);
        t.pop_back();
    }
}
vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
    G = graph; n = G.size();
    t.push_back(0);
    DFS(0);
    return res;
}
```

## 721. 账户合并

这道题我们需要积累几个代码技巧：
1. 需要用 hash 表：将字符串 与 并查集中的节点 绑定在一起
2. 需要给 `每一个` `集合代表`(即一个代表性节点) `创立`一个 性质数组(存储一个`合并后`的账户)
3. 这个性质数组中的每一个元素代表了合并后的一个集合，也就是 集合代表所代表的集合。

遍历技巧：

并查集算法的代码分析技巧就是：分`两次`对 `原离散`集合 进行遍历：
1. 第一次遍历是为了：进行合并相同的元素。
2. 第二次是为了：给每一个 `集合代表` 创建性质数组。
3. 第三次遍历即使为了创建答案，即 当 px == x 时，即 `处于` `集合代表时`，进行

这题`难点`在于它是一个类似 `两重` 并查集的模式：
1. `第一层`并查集在一个 accounts[i] 内，所有的`邮箱地址`都会映射到一个`数字编号 i` 
2. `第二层`并查集就是我们常用的并查集，即`数字 i` 会映射到一个集合代表：`px = find(x)`

这种 `两层映射` 就是像双层并查集一样，是有难度的，我们需要通过判断 `最底层` 的 邮箱地址 `是否曾经出现过` 来决定 它能否在 `第二层(即编号层)` 进行一个 Merge 操作。抽象来说即判断两个节点之间是否有连接边：

如何判断`编号节点`之间是否存在连接边：`用抽象的概念来说`：我们如何判断 `两个 节点(数字编号)` 之间是否存在一条边，是通过 判断`他们之间是否存在一条边`：即 `存在邮箱地址相同` -> 转化为 `曾经出现过`相同的邮箱地址

结合代码进行分析：

```c++
int p[N];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
void merge(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return;
    p[px] = py;
}
unordered_map<string, int> mp;
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
    int n = accounts.size();
    for (int i = 0; i < n; i ++ ) p[i] = i;

    for (int i = 0; i < n; i ++ ) 
        for (int j = 1; j < accounts[i].size(); j ++ ) 
            if (!mp.count(accounts[i][j]))      // 判断 `编号节点` 之间是否存在连接边
                mp[accounts[i][j]] = i;
            else        // 如果曾经出现过 相同邮箱地址，则 i 与 mp[accounts[i][j]] 存在一条连接边
                merge(mp[accounts[i][j]], i);
    
    vector<set<string> > arr(n);
    for (int x = 0; x < n; x ++ ) {         // 构造 性质数组，即对每个 集合代表 创建一个邮箱集合 set  
        int px = find(x);
        for (int j = 1; j < accounts[x].size(); j ++ )
            arr[px].insert(accounts[x][j]);
    }
    vector<vector<string>> res;
    for (int i = 0; i < n; i ++ ) {     // 通过 x == px 来构建 答案
        int pi = find(i);
        if (pi == i) {
            vector<string> t;
            t.push_back(accounts[pi][0]);
            for (auto e : arr[pi]) t.push_back(e);
            res.push_back(t);
        }
    }
    return res;
}
```



## 299. 猜数字游戏
此题是 `阅读理解 + 数学` 即我们需要的是求出可以匹配的数(数字与位置都相同)，以及多少`可以匹配（即数字相同位置不同）`，但不在匹配位置数字的数量

直接上答案的代码：
```c++
string getHint(string s1, string s2) {
    int cnt1[10], cnt2[10], cnt3[10];
    memset(cnt1, 0, sizeof cnt1), memset(cnt2, 0, sizeof cnt2), memset(cnt3, 0, sizeof cnt3);
    int n = s1.size();
    int x = 0, y = 0;
    for (int i = 0; i < n; i ++ ) 
        if (s1[i] == s2[i]) 
            x ++ ;
        else 
            cnt1[s1[i] - '0'] ++ , cnt2[s2[i] - '0'] ++ ;
    
    for (int i = 0; i < 10; i ++ ) {
        int t = min(cnt1[i], cnt2[i]);
        y += t;
    }
    return to_string(x) + 'A' + to_string(y) + 'B';
}
```

## 普通 BFS 与 SPFA 之间的联系：

1. 适用性层面：SPFA 适用于 `边权任意` 的 最短路算法，而 BFS `只`适用于 `边权为 1` 的算法
2. `v[] 数组与 d[] 数组的区别与联系`：SPFA 的 d[] 数组，即距离数组，`充当`了 BFS 中 的 v[] 数组。即 BFS 是只要访问过了一个节点，那么该节点
2. 使用场合层面：SPFA 只要能够将图中的 d[] 数组`通过题意构造好`，那么它 一定能代替 BFS。所以我们必须巧妙地转化题意，构造正确的 d[] 数组

题目推荐：
### LeetCode 934 最短的桥
mat[][] 数组转化为 -1，0，1； 超级源点(即多源 BFS, 将多个原点的 d[srci] 置为 0 ，并将多个源点进行入队)


### 1129. 颜色交替的最短路径
：`建立多图` + `搜索多状态`

此题与 `ACwing 3480. 棋盘游戏` 只能说思想一模一样，都是用两个状态来进行搜索，也就是：拓展 d[] 数组的维度，将另一个维度扩展为状态维度。然后进行 SPFA 来做一个最短路的搜索：


`关键点`：
1. inq[], d[], 等等都要设置成为多维度才能成功
2. 我们搜索的最短路一定要从 `节点 与 状态` 两个维度方面进行。`缺一不可`

### 417. 太平洋大西洋水流问题
所以说，写 BFS 的题都是要 `深刻理解好 v[] 数组和 d[] 数组` 就能写好这种题

经典的超级源点 BFS，利用边界节点和 `深刻理解 v[] 数组` 的含义来写出这道题：令 `f[] 数组` 为从左上侧进行 BFS 的 v[] 数组， `g[] 数组`为从右下侧进行 BFS 的 v[] 数组，那么最终 对于一个节点来说，f[x] == g[x] == 1 才能说明他们是符合题意的节点

代码如下：
```c++
const int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
vector<vector<int>> pacificAtlantic(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    int f[m][n], g[m][n]; memset(f, 0, sizeof f), memset(g, 0, sizeof g);
    queue<PII> q;
    for (int j = 0; j < n; j ++ ) q.push({0, j}), f[0][j] = 1;
    for (int i = 1; i < m; i ++ ) q.push({i, 0}), f[i][0] = 1;
    while (q.size()) {
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 4; i ++ ) {
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex < 0 || nex >= m || ney < 0 || ney >= n || mat[nex][ney] < mat[x][y] || f[nex][ney]) continue;
            q.push({nex, ney}), f[nex][ney] = 1;
        }
    }
    
    for (int j = 0; j < n; j ++ ) q.push({m - 1, j}), g[m - 1][j] = 1;
    for (int i = m - 2; i >= 0; i -- ) q.push({i, n - 1}), g[i][n - 1] = 1;
    while (q.size()) {
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 4; i ++ ) {
            int nex = x + dirs[i][0], ney = y + dirs[i][1];
            if (nex < 0 || nex >= m || ney < 0 || ney >= n || mat[nex][ney] < mat[x][y] || g[nex][ney]) continue;
            q.push({nex, ney}), g[nex][ney] = 1;
        }
    }
    
    vector<vector<int> > res;
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ )
            if (g[i][j] == 1 && g[i][j] == f[i][j]) 
                res.push_back({i, j});
    return res;
}
```

## 建树与建图技巧：反向建边，建立有向无环图，建立有向树等，题目推荐：前面的 洛谷题目的反向建边 也很推荐


###  802. 找到最终的安全状态：
通过反向建边，并将最后 `ind[x] == 0` 的节点判为安全数组。也就是`加入 top 序列的数组的节点是 安全节点`。原因：如果一个节点`所有出边`都是安全的，那么这个节点也是安全的


### 1466. 重新规划路线
这题是可以通过`分别建两个图`并且要用反向建边方式 x -> y 建成 x <- y ， x <- y 建成 x -> y，然后通过一次 DFS 来记录在原 x -> y (即建成 x <- y) 的边会经过几条。答案就是这些经过的边的边数。但是答案是通过`对边进行分类`来写的。可以参考答案的写法。

### 1376. 通知所有员工所需的时间
我们通过建立一个有向树，来自顶向下进行 DFS 即可求出答案。

## 在图上进行 DP

### 2050. 并行课程 III
经典的求关键路径题目，关键路径就是用 DP 来做的

### 1857. 有向图中最大颜色值
我们令 f[x][j] 代表 表示以节点 x 为终点的`所有路径`中，包含颜色 j 的节点数量的最大值。

在这些图上 DP 问题中，我们总是定义为 ： 以 x 为终点，而不是以 x 为起点，这是一个小技巧

那么我们可以写出动态规划方程：
1. 初始化 `f[x][colors[x]] = 1` 代表当前节点自己的颜色时，只有该颜色的数量为 1 .
2. `f[y][j] = max(f[y][j], f[x][j]) ： 当 colors[y] != j 时`；
3. `f[y][i] = max(f[y][i], f[x][i] + 1) ： 当 colors[y] == j 时` 

或者可以这样：
1. 初始化所有的 `f[][] = 0` 
2. 转移方程：`f[y][j] = max(f[y][j], f[x][j])`
3. 在每次选中一个节点时(即开头的：`int x = q.front(); q.pop()`)，对 f[x][colors[x]] ++ ;

代码：
```c++
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[N];
void insert(int x, int y) {
    G[x].push_back(y); ind[y] ++ ;
}
int f[N][26], ind[N];
int n;
string colors;
int topsort(){
    queue<int> q;
    for (int i = 0; i < n; i ++ )
        if (ind[i] == 0) q.push(i);
    int cnt = 0;
    while (q.size()) {
        int x = q.front(); q.pop();
        cnt ++ ;
        f[x][colors[x] - 'a'] ++ ;
        for (auto [y] : G[x]) {
            for (int i = 0; i < 26; i ++ )     // DP
                f[y][i] = max(f[y][i], f[x][i]);
            if ( -- ind[y] == 0) q.push(y);     // 入队
        }    
    }
    if (cnt != n) return 0;
    return 1;
}
int largestPathValue(string _colors, vector<vector<int>>& edges) {
    colors = _colors;
    n = colors.size();
    memset(f, 0, sizeof f);
    for (auto e : edges) {
        int x = e[0], y = e[1];
        insert(x, y);
    }
    if (!topsort()) return -1;

    int ans = 1;
    for (int i = 0; i < n; i ++ ) 
        for (int j = 0; j < 26; j ++ )
            ans = max(ans, f[i][j]);
    return ans;
}
```

## 并查集 + 连通图题目推荐

### 2316. 统计无向图中无法互相到达点对数`：`连通图  + 数学
可以利用 `并查集` 或者 `DFS` 来处理连通图的这一部分，然后利用 迭代数学来求出解

## kruskal 与 prim 算法推荐：前面的 洛谷题单 很推荐


## 1140. 石子游戏 II
由于我自己的原因，我自己总是没有认真地去处理边界，导致总是在边界的问题上犯错。

代码技巧：

如果前缀和用从 1 ~ n 表示，那么你最好用 1 ~ n 的范围来表示原数组

如果你的原数组为 0 ~ n - 1，则前缀和别忘了可能要用 s[i + 1] 或者 s[j + 1] 来表示，而不是 s[i] 和 s[j]

`代码简洁性分析`：到底用`长度`遍历还是用`结尾下标`遍历数组？
1. 如果题目的要求与长度有关，则最好用长度遍历
2. 如果题目要求与下标有关，则最好用下标遍历

代码如下：

```c++
vector<int> nums;
int memo[110][110];
int s[110];
int n;
int DFS(int i, int M) {
    if (i == n) return 0;
    if (memo[i][M] != -1) return memo[i][M];
    int &res = memo[i][M], sum = 0; res = 0;
    for (int x = 1; x <= 2 * M && i + x - 1 < n; x ++ ) {       // 如果题目的要求与长度有关，则最好用长度遍历
        sum += nums[i + x - 1];         // 这里可以简化算式，注意到： sum + s[n - 1] - s[i + x - 1] == s[n - 1] - s[i - 1] 但是这里要注意边界。因为 i - 1 必须 >= 0。所以最好还是用 1 ~ n 来存储前缀和，而这里是 0 ~ n - 1 ，有点不好。
        res = max(res, sum + s[n - 1] - s[i + x - 1] - DFS(i + x, min(n, max(M, x))));
    }
    return res;
}
int stoneGameII(vector<int>& _nums) {
    nums = _nums; n = nums.size();
    memset(memo, -1, sizeof memo), memset(s, 0, sizeof s);
    s[0] = nums[0];
    for (int i = 1; i < n; i ++ ) 
        s[i] = s[i - 1] + nums[i];
    return DFS(0, 1);
}
```

## 1006. 笨阶乘
这题实在是太妙了，这个栈模拟实在想不到啊。

由于他这个 `* / + -` 需要考虑到运算优先级，所以我们只能从左往右处理，但是我们如何进行处理运算优先级的关系呢？

首先，`使用双栈的通用方法`一定是可行的，因为它可以处理含括号的表达式。

但是我们可以用一个更简单的方式来进行处理：`这种方法`可以处理`不含括号`的表达式

1. 如果是 `* /` -> 我们直接对 `栈顶 back()` 和 `栈外元素 n` 进行运算后直接入栈
2. 如果是 `+ -` -> 我们直接将栈外元素入栈，注意入栈时的正负号。

最后对栈内所有元素进行求和即可

```c++
int clumsy(int n) {
    vector<int> stk;
    stk.push_back(n);
    int mode = 0;
    while ( -- n) {
        if (mode == 0) stk.back() *= n;     // 如果是 `* /` -> 我们直接对 `栈顶 back()` 和 `栈外元素 n` 进行运算后直接入栈
        else if (mode == 1) stk.back() /= n;
        else if (mode == 2) stk.push_back(n);       // 如果是 `+ -` -> 我们直接将栈外元素入栈，注意入栈时的正负号。
        else stk.push_back(-n);     // 注意这里是将 -n 入栈
        mode = (mode + 1) % 4;
    }
    int ans = 0;
    while (stk.size()) {    // 最后对栈内所有元素进行求和即可
        ans += stk.back();
        stk.pop_back();
    }
    return ans;
}
```

## 1041. 困于环中的机器人
此题为：`规律 + 模拟` 

方法一：找规律
规律为： 
1. 如果走完一次指令后仍处于原点，则一定处于环中
2. 如果走完一次指令后，它不处于原点，且方向不为初始方向，则它一定会重新返回原点。原理：
2.1 > 如果执行完一次指令后方向朝南则下一次一定会回到原点，方向朝北，即走了一次负的位移。即`执行两次`指令后回到原点
2.2 > 如果执行完一次指令之后，机器人朝东，则下一次指令又会使机器人右转，变为朝南，则变为 2.1 的情况，会走负位移回到 原点。也就是`执行四次`指令后回到原点

总结：如果执行四次指令后回到原点，则说明一定有环：

```c++
const int dirs[4][2] = { {-1, 0}, {0, 1}, {1, 0}, {0, -1} };
bool isRobotBounded(string s) {
    s = s + s + s + s;
    int dir = 0, n = s.size(), x = 0, y = 0;
    for (int i = 0; i < n; i ++ ) {
        if (s[i] == 'G') x += dirs[dir][0], y += dirs[dir][1];
        else if (s[i] == 'R') dir = (dir + 1) % 4;
        else if (s[i] == 'L') dir = (dir + 3) % 4;
    }
    return x == 0 && y == 0;
}
```

方法二：增加指令长度到 4*n，然后判断是否处于原点：这个就是增加偶数循环次数来搞他，很爽的


## 223. 矩形面积
这是一道很好的模拟题：`容斥原理 + 数学几何`

脑海图：记住左下角和右上角这两个坐标，标识了一个唯一的矩阵

分两种情况讨论：
1. 完全不相交：此时返回两个矩形的面积之和：S1 + S2
2. 有重叠的部分：计算 `重叠部分的矩形为：` 左下角选择最靠右上的；右上角选择最靠左下的。

代码如下：
```c++
int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
    int S1 = (ax2 - ax1) * (ay2 - ay1), S2 = (bx2 - bx1) * (by2 - by1);
    if (ax2 <= bx1 || bx2 <= ax1 || ay1 >= by2 || by1 >= ay2) return S1 + S2;       // 完全不相交的情况
    int cx1 = max(bx1, ax1), cy1 = max(by1, ay1);       // 重叠部分的 左下角选择最靠右上的
    int cx2 = min(ax2, bx2), cy2 = min(ay2, by2);       // 重叠部分的 右上角选择最靠坐下的
    return S1 + S2 - (cx2 - cx1) * (cy2 - cy1);     // 容斥原理计算答案
}
```

## 1094. 拼车
典型的差分数组应用题,

这题直接上代码，后面比较有意思的差分题目再上解析：

```c++
int a[1010];
bool carPooling(vector<vector<int>>& trips, int capacity) {
    memset(a, 0, sizeof a);
    for (auto e : trips) {
        int from = e[1], to = e[2], cnt = e[0];
        a[from] += cnt, a[to] -= cnt;       // 我们会在 to 这个点下车，所以增加乘客量的的区间是 [fronm, to - 1] 
    }
    if (a[0] > capacity) return 0;
    for (int i = 1; i <= 1000; i ++ ) {
        a[i] = a[i - 1] + a[i];
        if (a[i] > capacity) return 0;
    }
    return 1;        
}
```
## 1109. 航班预订统计

这题遇上一题 `1094. 拼车` 同理，直接上代码：

```c++
void insert(int l, int r, int inc) {
    b[l] += inc, b[r + 1] -= inc;
}
int b[20100];
void insert(int l, int r, int c) {
    b[l] += c, b[r + 1] -= c;
}
vector<int> corpFlightBookings(vector<vector<int>>& books, int n) {
    memset(b, 0, sizeof b);
    for (auto e : books) {
        int l = e[0], r = e[1], cnt = e[2];
        insert(l, r, cnt);
    }
    vector<int> res;
    int sum = 0;
    for (int i = 1; i <= n; i ++ ) {
        sum += b[i];
        res.push_back(sum);
    }
    return res;
}
```

## 2129. 将标题首字母大写
c++ 内 `toupper(char) 与 tolower(char) 函数` 的学习：

返回一个将 字母参数变为 大写或者小写 的函数，可使代码更简洁

代码如下：这里分享以下灵神的代码，因为它不用借助 split() 函数，stringstream 是`专门`针对 `空格分隔符` 的可替代 split() 函数的一种方法
```c++
string capitalizeTitle(string title) {
    istringstream iss(title);
    string ans, s;
    while (iss >> s) {
        if (!ans.empty()) 
            ans += ' ';
        
        if (s.length() > 2) 
            ans += toupper(s[0]),   // 将首字母加入答案
            s = s.substr(1);        // 将首字母删除
        
        for (char c : s)    // 将非首字母加入答案
            ans += tolower(c);
        
    }
    return ans;
}
```

## 1218. 最长定差子序列
该题可以`用最长递增子序列`的思想来做：但是这种方法超时了：令 f[i] 为 以 i 为结尾的，公差为 dif 的最长子序列，那么转移方程为 ：

`f[i] = max(f[i], f[j] + 1) 其中需满足 arr[i] - arr[j] = dif`

而这种方法是经典的 O(n^2) 会超时，有什么方法优化一下呢？

我们记 `(i, val)` 为 在 [0...i] 内的 `所有以 值 val 为结尾的子序列`。

记 `f(i, val)` 为以上所有子序列中长度最长的子序列的长度

空间优化：由于 f(i) 一定要包含所有的 f(i - 1) 所以我们省略掉第一个维度。

代码如下：

```c++
int longestSubsequence(vector<int>& arr, int d) {
    int n = arr.size();
    int ans = 1;
    unordered_map<int, int> f;
    for (int i = 0; i < n; i ++ ) {
        f[arr[i]] = max(f[arr[i]], f[arr[i] - d] + 1);
        ans = max(ans, f[arr[i]]);
    } 
    return ans;
}
```

## 最长递增子序列题目总结：

### 673. 最长递增子序列的个数
该题和 Dij 求最短路的个数一样，是一个经典的转移方法：
1. 如果当前这个位置 x 的最短路(或最长递增子序列) 是由某一个位置 y 转移过来的（即被 y 的状态更新了），则路径个数就从 y 那里复制过来： `cnt[x] = cnt[y]`
2. 如果当前位置 x 的最短路与 y 转移过来的路径长度相同，则加上 y 那里的路径个数：cnt[x] += cnt[y]
3. 否则不对路径个数这个状态进行更新。

### 354. 俄罗斯套娃信封问题 
这个题就是有点难度了，需要在第一个维度进行升序，即可以使得下一封能装入上一封信。同时，他又需要在第二个维度进行降序，因为这样可以省略第一个维度的判断，即不需要判断是否第一个维度相等。

如果第二个维度不降序，则会导致 [1, 1], [1, 2] 这种情况下 让答案误认为 2，但实际上却是 1。本质上来说：第二维度`逆序排序`保证在 w 相同的信封中`最多``只选`取一个。

### 合法性递增子序列问题：
这种题目非常典型，我们只能去枚举那些合法的递增子序列，即能够转移过来的子序列，而那些非法的子序列我们无法进行转移。典型的题目有：`两个键的键盘`，`823. 带因子的二叉树` 我们都是只能在 arr[j] 为 arr[i] 的因子时，才能合法的转移过来，其他的情况则转移不了。

## 最长公共子序列问题 & 字符串 DP 问题：

### 1035. 不相交的线：本质上就是最长公共子序列
可以从 `几何` 上来分析如果`想要不相交`，则`一定是公共子序列`

转移方程为：`f[i][j] = nums1[i - 1] == nums2[j - 1] ? f[i - 1][j - 1] + 1 : max(f[i - 1][j], f[i][j - 1])`

### 1312. 让字符串成为回文串的最少插入次数
直接上转移方程：

```c++
if(s[i] == s[j]) dp[i][j] = dp[i + 1][j - 1];
else dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1);
```

### 最长回文子序列：
```c++
f[l][r] = s[l - 1] == s[r - 1] ? f[l + 1][r - 1] + 2 : max(f[l + 1][r], f[l][r - 1]);
```

### 72. 编辑距离
```c++
if (s1[i - 1] == s2[j - 1]) f[i][j] = f[i - 1][j - 1];
else f[i][j] = min({f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]}) + 1;
```

### 712. 两个字符串的最小ASCII删除和
```c++
if (s1[i - 1] == s2[j - 1]) f[i][j] = f[i - 1][j - 1];
else f[i][j] = min(f[i - 1][j] + s1[i - 1], f[i][j - 1] + s2[j - 1]);
```


## 背包问题：

### 474. 一和零
这是一个单物品维度，`双体积维度` 的背包问题，所以很有学习价值，经过状态压缩后的状态转移方程如下：

值得注意的是，这里的维度需要考虑当体积为 0 的情况。因为其中可以有一个维度的体积为 0.
```c++
f[j][k] = max(f[j][k], f[j - num0][k - num1] + 1);
```

### 1155. 掷骰子等于目标和的方法数
遇上一题不同，我们这里除了 i = 0 的情况下可以让 f[0][0] 为 0，i != 0 时，f[i][0] 一定是不为 0 的，这是从逻辑的层面来考虑的。

#### 如何预处理构造背包？
1. 构造容积(体积)
```c++
// 阶乘作为容积：
for (int i = 0, t = 1; t <= n; i ++ ) 
    v.push_back(t), t *= i + 1;

// 完全平方数作为容积：
for (int i = 0, t = 1; t <= n; i ++ )
    v.push_back(t), t = (i + 1) * (i + 1);

```
2. 构造价值：

这个`价值`就不能用常理揣测了，其`具体情况应该具体分析`：
1. `是否`能构造成功  -> bool
2. `总`的构造方案数  -> f[j] += f[j - v[i]]
3. 能够得到的价值(最小 or 最大)  -> f[j] = max(f[j], f[j - v[i]] + w[i])

## 96. 不同的二叉搜索树
这题我建议直接去看`代码随想录`的视频，不然理解不了：https://www.bilibili.com/video/BV1eK411o7QA/?spm_id_from=333.337.search-card.all.click&vd_source=7b6afcc85bf79519cec2f6f69c4c54fc

## 2140. 解决智力问题
这题与 `983. 最低票价` 简直一模一样。我们直接利用记忆化搜索就可以实现，直接上代码：
```c++

```


## 对于 `背包` 问题内的 `记忆化搜索形式` 与 `递推迭代形式` 的理解与分析
对于一个记忆化搜索的函数签名来说，每`多一个参数`，则减少一次对应的 `循环` 递推：

例：
1. 如果函数签名为 `DFS(i, j)` 则这个函数中将一个循环也没有。
2. 如果函数签名为 `DFS(i)` 或者 `DFS(j)` 函数中，只有一个循环

原因：从数学角度分析，每对一个参数进行 `DFS 深度优先搜索` ，那么它会自动地帮你完成其中一个循环，即 `以递归栈` 的形式帮你完成一个 `循环递推`的 迭代模式。

## 2466. 统计构造好字符串的方案数
经典的 `容积在外`，`物品在内` 的背包问题，也就是说物品之间可以进行排列。有顺序区分。

`注`：很多人说这是跳台阶的变形题目，但是我认为`跳台阶`正是`这种（容积在外，物品在内）`背包问题的一个特殊案例，所以背包比跳台阶更加具有一般性与普遍性

直接上代码：
```c++
const int MOD = 1e9 + 7;
typedef long long LL;
LL f[N];
int countGoodStrings(int low, int high, int zero, int one) {
    LL v[2] = {zero, one};
    memset(f, 0, sizeof f); f[0] = 1;
    for (int j = 1; j <= high; j ++ )
        for (int i = 0; i < 2; i ++ )
            if (j >= v[i])
                f[j] = (f[j] + f[j - v[i]]) % MOD;
    LL ans = 0;
    for (int j = low; j <= high; j ++ )
        ans = (f[j] + ans) % MOD;
    return ans;
}
```

## 79. 单词搜索  ->  养成良好的 DFS 习惯
注：如果你能够养成良好的 DFS 习惯，那么你 Debug 的时间将会减少很多：

良好习惯：
1. 不可往回搜索：我们必须设置一个 `v[] 数组` 来确保 DFS 不会朝来时的路径上返回去进行搜索，这样会导致 `爆栈`
2. 在进入递归函数 `之前` 去判断是否剪枝：为了使代码具有简洁优雅的特性，我们应该在进入 DFS() 之前，去判断 我们`生成的状态` 是否要进行 DFS ，而`不是`在`已经` 进入了 `当前状态` 的 DFS() 后才进行剪枝判断。
3. 在进入 DFS() 之前，将路径（即生成的状态）加入 t(state)  中，并 `同时` 进行 v[t] = 1.
4. 在离开 DFS() 之后，将上次生成的状态从 t(state) 中移除，并 `同时` 进行 v[t] = 0.

那么这就是 DFS 的好习惯，希望你能永远保持下去！

`好习惯` DFS 代码如下：
```c++
bool flag;
string tar;
string t;
int m, n;
vector<vector<char>> mat;
int v[7][7];
const int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
void DFS(int x, int y) {
    if (flag) return;
    if (t.size() == tar.size()) {
        flag = 1; return;
    }
    for (int i = 0; i < 4; i ++ ) {
        int nex = x + dirs[i][0], ney = y + dirs[i][1];
        if (nex < 0 || nex >= m || ney < 0 || ney >= n || v[nex][ney]) continue;
        if (mat[nex][ney] != tar[t.size()]) continue;
        t.push_back(mat[nex][ney]), v[nex][ney] = 1;
        DFS(nex, ney);
        t.pop_back(), v[nex][ney] = 0;
    }
}
bool exist(vector<vector<char>>& _mat, string _tar) {
    mat = _mat; memset(v, 0, sizeof v);
    m = mat.size(), n = mat[0].size(); 
    tar = _tar; flag = 0;
    for (int i = 0; i < m; i ++ ) {
        for (int j = 0; j < n; j ++ ) {
            if (mat[i][j] != tar[0]) continue;
            t.push_back(mat[i][j]), v[i][j] = 1;
            DFS(i, j);
            t.pop_back(), v[i][j] = 0;
            if (flag == 1) break;
        }
    }
    return flag;
}
```

## 274. H 指数
方法 1 ： 二分查找  (略)

方法 2 ： 排序 + 遍历  (略)

方法 3 ： 计数
我们用 `cnt[] 数组` 去统计论文的引用次数 cnt，然后从 n -> 1 来遍历整个 cnt 数组，直到有 Σcnt[i] 能够 >= i 即可。

```c++
int hIndex(vector<int>& nums) {
    int n = nums.size();
    vector<int> cnt(n + 1);
    for (int e : nums) cnt[min(e, n)] ++ ;
    int s = 0;
    for (int i = n; ; i -- ) {
        s += cnt[i];
        if (s >= i) return i;
    }
}
```


## 134. 加油站

暴力写法：如果`连暴力模拟都不会还优化个屁`：
```c++
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int n = gas.size();
    for (int start = 0; start < n; start ++ ) {     // 枚举每一个起点
        int flag = 1;
        for (int i = start, t = 0; i < n + start; i ++ ) {
            t += gas[i % n] - cost[i % n];
            if (t < 0) { flag = 0; break; }     // 如果当前 油 不够了，则 进行下一次 start 的判断。
        }
        if (flag == 1) return start;    // 如果能够走完所有的位置，则返回答案
    }
    return -1;
}
```

那么我们来看一下贪心写法：

根据`折线图`，我们必须 选择 `净亏损最严重` 的那一天`之后`的那天 作为我们选择 的答案，也就是`开车的起点`
```c++
int canCompleteCircuit(std::vector<int>& gas, std::vector<int>& cost) {
    int n = gas.size();
    int sum = 0;
    int minSum = INT_MAX, ans = 0;

    for (int i = 0; i < n; i++){
        sum += gas[i] - cost[i];    // sum 可以看作是 当前 净亏损值

        if (sum < minSum)      // 更新 最低净亏损值 ，并记录下这一个 位置
            ans = i,    
            minSum = sum;
        
    }
    if (sum < 0) return -1;     // 如果 sum < 0 则一定不可能 走完所有位置
    else if (minSum >= 0) return 0;     // 如果 净亏损最低值 >= 0 则在 任何 位置都可以 作为 起点答案
    else return (ans + 1) % n;      // 如果 净亏损最低值
}
```
## 436. 寻找右区间
正常做法：
```c++
vector<int> findRightInterval(vector<vector<int>>& itvs) {
    int n = itvs.size();
    vector<int> res(n, -1);
    for (int i = 0; i < n; i ++ ) {
        int l = itvs[i][0], r = itvs[i][1];
        for (int j = 0; j < n; j ++ ) {
            int lt = itvs[j][0], rt = itvs[j][1];   
            if (lt >= r && (res[i] == -1 || itvs[res[i]][0] > lt)) res[i] = j;      // 这个判断很想 Dij 的 选点 判断语句
        }
    }
    return res;
}
```

二分优化：
我们记录下每个区间的下标 id 以及他们的左区间，然后进行 二分搜索
```c++
vector<int> findRightInterval(vector<vector<int>>& itvs) {
    int n = itvs.size();
    vector<int> res(n, -1);
    vector<PII> _itvs;
    for (int i = 0; i < n; i ++ ) 
        _itvs.push_back({itvs[i][0], i});   // _itvs 是经过排序后的区间数组
    sort(_itvs.begin(), _itvs.end());

    for (int i = 0; i < n; i ++ ) {
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (_itvs[mid].first >= itvs[i][1]) r = mid;
            else l = mid + 1;
        }
        if (_itvs[l].first >= itvs[i][1]) res[i] = _itvs[l].second;
    }
    return res;
}
```

## 188. 买卖股票的最佳时机 IV
直接上代码体会：

关于这里的`边界问题`：
1. 由表达式可知， f[i][1][1] 一定等于 -nums[i - 1] 即进行交易一次后的值。如果 `状态 1 ` 在 `一次都不交易` 的情况下，那么它 一定是 -INF。`这代表`：`不交易的话 ，状态 1 是非法状态`，只有`至少交易一次`，`状态 1 才是合法状态`
2. 在 i == 0 时，即`股票还没有上线系统时`，所有的 状态 1 都是非法状态，因为在股票没上线时，一定不会持有股票。而 0 状态是初始化为 0，代表股票系统没上线时，其不持股状态的最大成交金额为 0.
3. 在对 第二个状态维度 进行迭代时，j 一定从 1 开始枚举，因为 j == 0 时，其值一定是初始化时的值，所以一定不要去迭代 j == 0 时的状态。而是从 1 开始进行迭代状态的值。
```c++
const int INF = 0x3f3f3f3f;
int maxProfit(int k, vector<int>& nums) {
    int n = nums.size();
    int f[n + 1][k + 1][2]; memset(f, 0, sizeof f);
    for (int j = 0; j <= k; j ++ )
        f[0][j][0] = 0, f[0][j][1] = -INF;      // 这个初始化一定非常重要，因为代表了没有交易时，其状态 1 都是非法状态
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= k; j ++ )      // j 一定从 1 开始枚举，因为 j = 0 时，答案一定是初始化的答案，一定不要去迭代 j == 0 时的状态
            f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1] + nums[i - 1]),
            f[i][j][1] = max(f[i - 1][j][1], f[i - 1][j - 1][0] - nums[i - 1]);     // 关于这里的边界问题：有表达式可知， f[i][1][1] 一定等于 -nums[i - 1] 即进行交易一次后， f[i][0][1] 原本是 0 ，即代表一次都不交易的话，那么它 一定是 -INF，即不交易的话 ，状态 1 是非法状态，只有至少交易一次，状态 1 才是合法状态
    return f[n][k][0];
}
```

## 309. 买卖股票的最佳时机含冷冻期
关键处：

1. 直接使用三种状态：不持股状态，持股状态，冷冻状态
2. 最后的答案要么在冷冻状态，要么在可以买股状态。绝对不是持股状态
```c++ 
const int INF = 0x3f3f3f3f;
int maxProfit(vector<int>& nums) {
    int n = nums.size();
    int f[n + 1][3]; memset(f, 0, sizeof f); 
    f[0][0] = 0, f[0][1] = -INF;        // 初始化
    for (int i = 1; i <= n; i ++ ) {
        f[i][0] = max(f[i - 1][0], f[i - 1][2]);
        f[i][1] = max(f[i - 1][1], f[i - 1][0] - nums[i - 1]);
        f[i][2] = f[i - 1][1] + nums[i - 1];
    }
    return max(f[n][0], f[n][2]);
}
```

## 1191. K 次串联后最大子数组之和
该题就是：先算出 k = 2 时的最大子数组和，再根据 `单个数组之和 sum` 是否 > 0 来递归判断是否要进行再次串联：
1. 因为如果 sum < 0 ，那么再次串联就会导致数组和不断减小
2. 如果 sum > 0 , 那么我继续串联，就会导致数组和不断增大。

代码如下：
```c++
int kConcatenationMaxSum(vector<int>& nums, int k) {
        if (nums.empty()) return 0;
        LL tmp = nums[0] > 0 ? nums[0] : 0LL;
        LL res = tmp, sum = 0;
        int n = nums.size();

        for (int i = 0; i < n; i ++ ) sum += nums[i];   // 计算单个数组 sum 的值

        for (int i = 1; i < min(k, 2) * n; i ++ ) {
            tmp = max(tmp + nums[i % n], (LL)nums[i % n]);      // 利用 O(1) 的空间优化来进行计算
            res = max(res, tmp);
        }

        while (sum > 0 && -- k >= 2)    // 如果 sum > 0 , 那么我继续串联，就会导致数组和不断增大。
            res = (res + sum) % MOD;

        return res % MOD; 
    }
```

## 1186. 删除一次得到子数组最大和
此题为`状态机DP`

这题要积累方法：状态定义和转移最重要：

f[i][0] 为没有进行删除过的，以 i 为结尾的最大子数组和；f[i][1] 为删除过一次的，以 i 为结尾的最大子数组和

代码：
```c++
const int INF = 0x3f3f3f3f;
int maximumSum(vector<int>& arr) {
    int n = arr.size(), ans = -INF;
    int f[n + 1][2]; memset(f, 0, sizeof f); f[0][0] = -INF, f[0][1] = -INF;
    for (int i = 1; i <= n; i ++ ) {
        f[i][0] = max(arr[i - 1], f[i - 1][0] + arr[i - 1]);
        f[i][1] = max(f[i - 1][1] + arr[i - 1], f[i - 1][0]);
        ans = max({ans, f[i][0], f[i][1]});
    }
    return ans;
}
```


## 789. 逃脱阻碍者


## 1277. 统计全为 1 的正方形子矩阵
这题只需要改变一下答案的累计即可：累计以 (i, j) 为结尾的所有`符合题意的正方形子集`

```c++
int countSquares(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    int f[m + 1][n + 1]; memset(f, 0, sizeof f);
    int ans = 0;
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ ) {
            if (mat[i - 1][j - 1] == 1) 
                f[i][j] = min({f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]}) + 1;
            ans += f[i][j];
        }
    return ans;
}
```

## 1048. 最长字符串链
这题的`答案思路`和我的思路一样，唯一的区别就是在比较`两个字符串是否为 前身 的关系`。我是用的是最长公共子序列模板，而答案使用的是 `哈希表 + 字符串截取`

我们都是从小到大枚举 字符串， 因为前身的关系就要求 长度只能相差 1。那么令 f(i) 为 以字符串 i 为结尾的词链中 最长的词链长度。

那么此题就转化为了`最长递增子序列类型`题目了.



答案代码如下：
```c++
bool cmp(const string &s1, const string &s2) {
    return s1.size() < s2.size();
}
int longestStrChain(vector<string>& words) {
    sort(words.begin(), words.end(), cmp);
    int ans = 0, n = words.size();
    unordered_map<string, int> mp;
    for (int i = 0; i < n; i ++ ) {
        int t = 1;
        for (int j = 0; j < words[i].size(); j ++ ) {   // words[i] 为第 i 个字符串
            string s = words[i].substr(0, j) + words[i].substr(j + 1);  // s 为被删除 1 个字符的字符串
            if (mp.count(s)) t = max(t, mp[s] + 1);
        }
        mp[words[i]] = t;
        ans = max(ans, t);
    }
    return ans;
}
```

## 313. 超级丑数
方法 1：最常用解法：`哈希表set + 最小堆`

方法 2：动态规划解法：

`定义` f(i) 为 第 i 个 超级抽数。那么根据题目要求： f(1) = 1

算法：
1. 我们对每一个 f[i] 即第 i 个丑数来说，通过遍历 primes[] 数组 与 p[] 指针数组 来 `获得第 i 个丑数`
2. 然后通过遍历 p 数组与 primes 数组来`对 p[] 数组进行更新`，即如果 f[p[j]] * primes[j] == f[i] ，则 j ++ ，将指针向右移动

```c++
typedef long long LL;
int nthSuperUglyNumber(int n, vector<int>& primes) {
    LL f[n + 1]; memset(f, 0x3f, sizeof f); f[1] = 1;
    int k = primes.size();
    vector<int> p(k, 1);


    for (int i = 2; i <= n; i ++ ) {    // 我们对每一个 f[i] 即第 i 个丑数来说
        
        for (int j = 0; j < k; j ++ )   // 通过遍历 primes[] 数组 与 p[] 指针数组 来 `获得第 i 个丑数`
            f[i] = min(f[i], f[p[j]] * primes[j]);

        for (int j = 0; j < k; j ++ )   // 然后通过遍历 p 数组与 primes 数组来 `对 p[] 数组进行更新`
            if (f[p[j]] * primes[j] == f[i]) p[j] ++ ;      // 如果 f[p[j]] * primes[j] == f[i] ，则 j ++ ，将指针向右移动
        
    }
    return (int)f[n];
}
```

## 1262. 可被三整除的最大和
状态机 DP ：定义 f(i, j) 为 从 nums[0...i] 中选取数字，能够得到的最大的数字之和 s，使其满足 s MOD 3 == j。那么我们定义其 `模三的余数 j `为状态。

那么我们考虑 第 i 个数，从`选或不选的角度`出发：
1. 如果该 nums[i] MOD 3 == 0, 那么 0 状态 应该从状态 0 转移过来：f(i, 0) = f(i - 1, 0) + nums[0]，且`一定要选`，1 状态同样从 1 状态转移过来，2 状态也从 2 状态转移过来。
2. 如果该 nums[i] MOD 3 == 1, 那么如果选 nums[i] 则 1 状态 应该从状态 0 转移过来；0 状态应该从 2 状态转移过来， 2 状态应该从  1 状态转移过来。不选的话就是从各自的上一个状态转移过来。
3. 如果 nums[i] MOD 3 == 2 , 那么同第 2 点的分析。

代码如下：
```c++
int maxSumDivThree(vector<int>& nums) {
    const int INF = 0x3f3f3f3f;
    int n = nums.size();
    vector<vector<int> > f(n + 1, vector<int>(3, -INF));
    f[0][0] = 0, f[0][1] = -INF, f[0][2] = -INF;        // 注意这里初始化一定要 将 1 和 2 状态的初始值设为 -INF
    for (int i = 1; i <= n; i ++ ) {
        if (nums[i - 1] % 3 == 0) {     // 如果 模 3 余 0 则 一定要选。不选就亏了一个数字之和
            f[i][0] = f[i - 1][0] + nums[i - 1];        
            f[i][1] = f[i - 1][1] + nums[i - 1];
            f[i][2] = f[i - 1][2] + nums[i - 1];
        }
        else if (nums[i - 1] % 3 == 1) {    // 如果是 余 1 则进行选或不选
            f[i][0] = max(f[i - 1][0], f[i - 1][2] + nums[i - 1]);
            f[i][1] = max(f[i - 1][1], f[i - 1][0] + nums[i - 1]);
            f[i][2] = max(f[i - 1][2], f[i - 1][1] + nums[i - 1]);
        }
        else if (nums[i - 1] % 3 == 2) {    // 如果是 余 2 同样进行选或不选 
            f[i][0] = max(f[i - 1][0], f[i - 1][1] + nums[i - 1]);
            f[i][1] = max(f[i - 1][1], f[i - 1][2] + nums[i - 1]);
            f[i][2] = max(f[i - 1][2], f[i - 1][0] + nums[i - 1]);
        }
    }
    return f[n][0];
}
```

代码简洁化：利用向后更新来使代码简洁化：`虽然代码简洁，但是不利于理解`
```c++
int maxSumDivThree(vector<int>& nums) {
    const int INF = 0x3f3f3f3f;
    int n = nums.size();
    vector<vector<int> > f(n + 1, vector<int>(3, -INF));
    f[0][0] = 0;
    f[0][1] = -INF;
    f[0][2] = -INF;
    for(int i = 0;i < n;i++)
        for(int j = 0;j < 3;j++)    // 我们这里利用向后更新来简化了状态转换的方程。但是不好理解
            f[i + 1][(j + nums[i]) % 3] = max(f[i][j] + nums[i],f[i][(j + nums[i]) % 3]);
        
    
    return f[n][0];
}
```

## 1227. 飞机座位分配概率
数学推导公式自己去看官方题解，代码略。

## 373. 查找和最小的 K 对数字
暴力写法：`暴力都不会还做个屁的优化`。

`误区`：如果我们一直`递增`地使用`贪心`方法来用`双指针`进行求解 k 个最小数对，这对于 k < nums1.size() + nums2.size() 的情况是正确的。因为`贪心 + 双指针`确实能够求出前面 k 个小数对。但是如果 k 足够大，则有些数对就被忽略了。即我们遍历完整个数组最多只能得到 n1 + n2 - 1 个最小数对，但是如果 k 大于 n1 + n2 - 1 ，那么我们这个做法就失效了。因为 nums1[] 与 nums2[] `总共` 可以有 `n1 * n2 个` 数对。

那么我们必须先求出所有的数对： n1 * n2 个数对，然后对其进行排序即可，代码如下：

```c++
bool cmp(const vector<int> &a, const vector<int> &b) {
    return a[0] + a[1] < b[0] + b[1];
}
vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
    int n1 = nums1.size(), n2 = nums2.size();
    vector<vector<int> > arr;
    for (int i = 0; i < n1; i ++ ) 
        for (int j = 0; j < n2; j ++ ) {
            arr.push_back({nums1[i], nums2[j]});
        }
    sort(arr.begin(), arr.end(), cmp);
    vector<vector<int> > res;
    for (int i = 0; i < arr.size() && i < k; i ++ ) {
        res.push_back(arr[i]);
    }
    return res;
}
```

最小堆优化：

`使用哈希表`：我们可以将下标为：(0, 0), (0, 1), (0, 2) ... (0, n2 - 1), (1, 0), (2, 0),  ... , (n1 - 1, 0) 这些加入小根堆中，再利用 `unordered_set` 来记录已经加入过的元素进行去重。

`不使用哈希表`：我们可以将关于 nums1[] 的所有下标先加入堆中`(0...n - 1, 0)` ，然后我们只对关于 nums2[] 维度的下标进行逐个增加，而对第一维度的下标固定不动。那么我们就可以`避免了有重复元素`，因为`对于每个坐标的第一个维度`来说，在堆中一定是唯一的，因为我们不会对第一个维度进行改变。这样我们将`不会`再`使用`到 `哈希表` 记录重复元素了。

`经典迭代模拟方法，生成元素数量迭代法`：利用所生成的 `数组元素数量个数` 进行 `遍历迭代` 。这点和 `54. 螺旋矩阵` 的模拟方法类似。

`写小根堆的好习惯`：
1. 在 比较函数的大括号前记得 `加 const`： const { ... }。即在写 operator 时，记住一定要有两个 const，`一个在小括号内`，`一个在大括号前`
2. 不要写成 greater<int> , 要写成 greater<Node>

代码如下：
```c++
struct Node {
    int i, j, sum;
    // bool operator < (const Node &other) const {
    //     return sum < other.sum;
    // }
    bool operator > (const Node &other) const {     // 在大括号前面忘记加 const 是最严重的错误： const {}
        return sum > other.sum;
    }
    Node() {}
    Node(int i, int j, int sum) : i(i), j(j), sum(sum) {}
};
vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
    priority_queue<Node, vector<Node>, greater<Node> > q;   // 这里的 greater<Node> 特别容易写成 greater<int>
    int n1 = nums1.size(), n2 = nums2.size();
    for (int i = 0; i < n1; i ++ ) q.push(Node(i, 0, nums1[i] + nums2[0]));
    vector<vector<int> > res;

    while (k -- && q.size()) {
        auto [i, j, sum] = q.top(); q.pop();
        res.push_back({nums1[i], nums2[j]});

        if (j + 1 < n2)     // 一定要对下标越界 非常 敏感
            q.push({i, j + 1, nums1[i] + nums2[j + 1]});
    }
    return res;
}
```

## 719. 找出第 K 小的数对距离
这是一个跟上一题很类似的题目，直接上代码，虽然 LeetCode 超时，但是可以用，我们`绝不做我们能力范围之外的优化`！

`宁愿更加暴力，也不愿做能力范围外的优化`

```c++
struct Node {
    int i, j, dif;
    Node() {}
    Node(int i, int j, int dif) : i(i), j(j), dif(dif) {}
    bool operator > (const Node &other) const {
        return dif > other.dif;
    }
};
int smallestDistancePair(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    priority_queue<Node, vector<Node>, greater<Node> > q;
    for (int i = 0; i < n - 1; i ++ ) 
        q.push({i, i + 1, abs(nums[i] - nums[i + 1])});
    int ans = 0;
    while (k -- && q.size()) {
        auto [i, j, dif] = q.top(); q.pop();
        ans = dif;
        if (j + 1 < n) 
            q.push({i, j + 1, abs(nums[i] - nums[j + 1])});
    }
    return ans;
}
```

## 648. 单词替换
此题和 `1048. 最长字符串链` 的模拟方法一样，经典的 `字符串截取 + 哈希表`

用 stringstream 进行处理由空格分割的字符串的用法：

```c++
// 正确用法：
stringstream iss(s);        // 一定要用这个方法构造！！！
vector<string> words;
string t;
while (s >> t) {
    words.push_back(t);
}

// 错误用法：
vector<string> words;
string t;
while (stringstream(s) >> t)    // 这样子写，将无限循环地输入 s 中 被分割出的第一个字符串！
    words.push_back(t);
```

代码如下：


```c++
string replaceWords(vector<string>& dictionary, string sentence) {
    unordered_set<string> S;
    for (auto e : dictionary) 
        S.insert(e);
    vector<string> words;
    string t;
    stringstream iss(sentence);
    while (iss >> t) words.push_back(t);
    
    for (auto &e : words) {
        for (int len = 1; len <= e.size(); len ++ ) {
            string sub = e.substr(0, len);
            if (S.count(sub)) {
                e = sub;
                break;
            }
        }
    }

    string ans;
    for (int i = 0; i < words.size(); i ++ ) {
        ans += words[i];
        if (i != words.size() - 1) ans += ' ';
    }
    return ans;
}
```
## 166. 分数到小数
首先我们来看 `高精度 除以 低精度` 的模板：
```c++
vector<int> div(vector<int> &a, int b, int &r) {
    vector<int> c;
    r = 0;
    for (int i = a.size() - 1; i >= 0; i -- ) {
        r = r * 10 + a[i];
        c.push_back(r / b);     // 我们可以保证 r / b < 10, 因为从数学上来说，(r % b) < b   =>    (r % b) * 10 < 10 * b        
                                //  =>    (r % b) * 10 / b < 10 
        r %= b;
    }
    reverse(c.begin(), c.end());
    while (c.size() && c.back() == 0) c.pop_back();
    return c;
}
```


那么我们直接上这个模拟代码：

```c++
string fractionToDecimal(int _x, int _y) {
    if (_y == 0) return "";
    if (_x == 0) return "0";
    string res;
    long long x = (LL)_x, y = (LL)_y; 
    if (x * y < 0) res += '-';      // 处理符号
    x = abs(x), y = abs(y);

    res += to_string(x / y); // 整数部分

    x %= y;
    if (x == 0) return res;
    
    res += '.';
    int index = res.size() - 1;
    unordered_map<int, int> mp;
    while (x && !mp.count(x)) {
        mp[x] = ++ index;
        x *= 10;                // 这三步的处理 非常像 高精度的模板。
        res += to_string(x / y);    // 我们可以保证 x / y < 10, 因为从数学上来说，(x % y) < y   =>    (x % y) * 10 < 10 * y        
                                    //  =>    (x % y) * 10 / y < 10    
        x %= y;
    }
    if (mp.count(x))
        res.insert(mp[x], "("), res += ')';
    return res;
}
```

## 820. 单词的压缩编码
这题是一个转换模拟题：
1. `哈希表 + 后缀逻辑` 

算法如下：
1. 我们用 set 去存储每个字符串。
2. 我们遍历每个字符串的后缀，如果该后缀存在于 set 中，则将该后缀从 set 中删除，这一步代表我们可以压缩该后缀编码
3. 我们将 set 中 从第 2 步删除后 剩余的字符串加入 结果中。

```c++
int minimumLengthEncoding(vector<string>& words) {
    unordered_set<string> S(words.begin(), words.end());
    for (string s : words) 
        for (int i = 1; i < s.size(); i ++ ) 
            S.erase(s.substr(i));
    int ans = 0;
    for (string s : S)
        ans += s.size() + 1;
    
    return ans;
}
```

## 752. 打开转盘锁

经典的 BFS 题目，养成 BFS 的好习惯：
1. 在 push 之前，先对 `v[]` or `d[]` 数组进行判断检查
2. 在 push 之后，对 `v[]` or `d[]` 数组进行 `记录` or `更新` 操作

3. 关于优先队列的 BFS ：同一 状态 可以 入队 多次，而 `以 同一状态 为更新源` 只能 `更新一次`，其他 多个相同的状态只能 continue ，即不能用来更新下一次的状态
4. 关于 SPFA 的 BFS：同一状态和以入队多次，也可以 更新其他状态多次。


```c++
unordered_set<string> dead, v;
string change(string x, int pos, int mode) {
    string res = x;
    if (mode == 0) 
        res[pos] = res[pos] == '9' ? '0' : res[pos] + 1;
    else 
        res[pos] = res[pos] == '0' ? '9' : res[pos] - 1;
    return res;
}
int openLock(vector<string>& deadends, string tar) {
    for (auto &e : deadends)
        dead.insert(e);
    queue<string> q;
    if (dead.count("0000")) return -1;
    q.push("0000"), v.insert("0000");
    int step = 0;
    while (q.size()) {
        for (int i = q.size(); i; i -- ) {
            auto x = q.front(); q.pop();
            if (x == tar) return step;
            for (int i = 0; i < 4; i ++ ) {
                string nex1 = change(x, i, 0), nex2 = change(x, i, 1);
                if (!dead.count(nex1) && !v.count(nex1)) q.push(nex1), v.insert(nex1);
                if (!dead.count(nex2) && !v.count(nex2)) q.push(nex2), v.insert(nex2);
            }
        }
        step ++ ;
    }
    return -1;
}
```

## 994. 腐烂的橘子
经典的 BFS 问题，我们 `利用 for 循环进行 层序遍历` 可以`不用`将 time 这个值 与 坐标值一起`构建`一个 `struct Node`， 减少了编码的复杂度

关键
1. 要养成对 v[] 数组进行明确定义的习惯，我们这里将 v[] 数组定义为腐烂的时间。如果 v[i][j] == -1 代表这个节点不会腐烂。
2. 我们可以利用 螺旋矩阵中的经典 `计数模拟法` 来判断是否 能够将所有的`新鲜橘子腐烂`，虽然这里的代码没有使用这个方法。

```c++
vector<vector<int> > v, mat;
const int dirs[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
int orangesRotting(vector<vector<int>>& _mat) {
    mat = _mat;
    int m = mat.size(), n = mat[0].size();
    v = vector<vector<int> >(m, vector<int>(n, -1));
    queue<PII> q;
    for (int i = 0; i < m; i ++ ) 
        for (int j = 0; j < n; j ++ ) {
            if (mat[i][j] == 2) q.push({i, j}), v[i][j] = 0;
        }
    int time = 0;
    while (q.size()) {
        time ++ ;
        for (int i = q.size(); i; i -- ) {
            auto [x, y] = q.front(); q.pop();
            for (int i = 0; i < 4; i ++ ) {
                int nex = x + dirs[i][0], ney = y + dirs[i][1];
                if (nex < 0 || nex >= m || ney < 0 || ney >= n || mat[nex][ney] == 0 || v[nex][ney] != -1) continue;
                q.push({nex, ney}), v[nex][ney] = time;
            }
        }
    }
    int ans = 0;
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ ) {
            if (mat[i][j] != 0 && v[i][j] == -1) return -1;
            ans = max(ans, v[i][j]);
        }
    return ans;
}
```
## 394. 字符串解码

在这题中，我们主要去区分：`步骤模拟` 与 `栈模拟` 的区别。

1. 步骤模拟举例：`592. 分数加减运算`, `640. 求解方程`
2. 栈模拟举例：`394. 字符串解码` , `71. 简化路径`

我们很容易通过观察发现，
1. `步骤模拟` 一定是遵循某个固定的顺序，在每一次大循环中去通过固定的顺序来模拟进行计算。
2. `栈模拟` 一定是去枚举每个字符，然后通过 该字符 来选择模拟不同的操作

栈模拟算法：
1. 创建 `数字栈 stk1` 存储 重复次数，创建 字符串栈 stk2 存储需要被重复的字符串，初始化栈中第一个元素为空字符串。我们将需要被重复的串简称为`重复串`，栈顶的重复串成为`栈顶串`
2. 循环遍历 s[i] 来进行 选择模拟不同操作：

2.1 > 如果为数字，则将数字部分加入 stk1

2.2 > 如果是 '[' ，则新建一个 `空重复串`，并将该 `空重复串` 加入 stk2，成为栈顶串。

2.3 > 如果是字符串，则将该字符串拼接入 stk2.back()

2.4 > 如果是 ']' ，则说明`完成了栈顶串的生成操作`。将栈顶串`进行自重复拼接`，并`弹出`，加入 stk2.back() 中。

```c++
string decodeString(string s) {
    int i = 0, n = s.size();
    vector<int> stk1;
    vector<string> stk2; stk2.push_back("");    // 初始化 res 为 空字符串
    while (i < n) {
        if (isdigit(s[i])) {
            int x = 0;
            while (i < n && isdigit(s[i]))
                x = x * 10 + s[i] - '0', i ++ ;
            stk1.push_back(x);
        }
        else if (s[i] == '[') stk2.push_back(""), i ++ ;
        else if (isalpha(s[i])) {
            string t;
            while (i < n && isalpha(s[i]))
                t += s[i], i ++ ;
            stk2.back() += t;
        }
        else {
            i ++ ;
            int repeat = stk1.back(); stk1.pop_back();
            string t = stk2.back(); stk2.pop_back();
            while (repeat -- ) stk2.back() += t;
        }
    }
    return stk2.back();
}
```


## 71. 简化路径
思路：先进行 split 划分，再用栈进行模拟：

```c++
vector<string> split(string s, char c) {
    vector<string> res;
    string t;
    for (auto e : s) 
        if (e == c)
            if (t.size()) res.push_back(t), t = "";
            else continue;
        else 
            t += e;
        
    if (t.size()) res.push_back(t);
    return res;
}
string simplifyPath(string s) {
    vector<string> stk1 = split(s, '/');
    vector<string> stk2;
    for (auto e : stk1) {
        if (e == ".") 
            continue;
        else if (e == "..")
            if (stk2.size()) stk2.pop_back();
            else continue;
        else 
            stk2.push_back(e);
    }
    string res = "/";
    for (int i = 0; i < stk2.size(); i ++ ) {
        res += stk2[i];
        if (i != stk2.size() - 1) res += '/';
    }
    return res;
}
```

## 820. 单词的压缩编码
利用 Tire 树的逆序建立来解决：
```c++
struct Node{
    unordered_map<int, Node*> son;
    int cnt;
    Node() { son.clear(), cnt = 0; }
};
Node *root = new Node();
void insert(string s) {
    int n = s.size();
    Node *cur = root;
    for (auto e : s) {
        if (cur->son[e] == NULL) cur->son[e] = new Node();
        cur = cur->son[e];
    }
    cur->cnt ++ ;
}
int ans = 0;
void DFS(Node *x, int depth) {      // 我们需要用 DFS 来统计压缩后的字符串长度。其形状和计算过程可以结合官网的图来进行理解，第一个 depth = 1 的
                                    // 初始化 相当于字符 '#'
    if (x->son.size() == 0) { ans += depth; return; }
    for (auto [k, y] : x->son) {
        DFS(y, depth + 1);
    }
}
int minimumLengthEncoding(vector<string>& words) {
    for (auto e : words) {
        reverse(e.begin(), e.end());
        insert(e);
    }
    DFS(root, 1);
    return ans;
}
```

## 421. 数组中两个数的最大异或值
直接暴力解决，不可能用 O(n) 的解法


## 131. 分割回文串
这题应该被划分在 子集回溯 类型中。因为假设我们要去分割这个字符串，那么我们就是在枚举 `分割位置子集` 即灵神所说的 `逗号子集` 。我们去枚举这个 `分割位置`，选或者不选 此处 进行分割。

实现：那么如何用代码实现这个思想呢？
1. 前提知识：DSF(idx) 代表的是：从 0 ... i 已经生成了一条合法路径，DFS(idx) 的目标是从 i >= idx 往后的位置去生成合法的路径
2. 我们从枚举答案的角度思考：枚举 j 从 idx ... n - 1 ：如果选第 j 个位置进行分割，那么我们就将从 idx ... j 处分割为一个回文串，如果不选，则 j 继续向后枚举。
3. 细节：如果我们枚举 j 到了 n - 1 都不选择分割，那么它就会跳出循环，即如果 n - 1 都不分割，那么无法成为一个合法答案。

回溯三问：
1. DFS(idx) 的含义是什么：DSF(idx) 代表的是：从 0 ... i `已经`生成了一条合法路径，DFS(idx) 的`目标`是从 i >= idx `往后的位置`去生成合法的路径
2. 当前操作？选或不选 + 合法性剪枝
3. 下一个子问题？从 下标 >= j + 1 后继续去构造合法路径：DFS(j + 1) 

那么我们就依此分析写出代码：
```c++
vector<vector<string> > res;
vector<string> t;
string s;
int n;
void DFS(int i) {
    if (i == n) { res.push_back(t); return; }

    for (int j = i; j < n; j ++ ) {     // 细节：如果我们枚举 j 到了 n - 1 都不选择分割，那么它就会跳出循环，即如果 n - 1 都不分割，那么无法成为一个合法答案
        string sub = s.substr(i, j - i + 1);        // 选或不选
        string subR = sub; reverse(subR.begin(), subR.end());   // 合法性剪枝
        if (subR != sub) continue;
        t.push_back(sub);
        DFS(j + 1);         // 下一个子问题？从 下标 >= j + 1 后继续去构造合法路径：DFS(j + 1) 
        t.pop_back();
    }
}

vector<vector<string>> partition(string _s) {
    s = _s;
    n = s.size();
    DFS(0);
    return res;
}

```


## 306. 累加数
跟 `131. 分割回文串` 一样，我们使用子集型分割，来对字符串进行 分割位置 的子集枚举：

难点：合法性剪枝：这里有两个很难的合法性剪枝操作：
1. sub 不为 0 时，不能有前缀 0
2. 如果路径长度已经 >= 2，则你需要保证结尾能够满足：路径的最后一个元素 + 倒数第二个元素 = `当前分割`出来的 s[idx ... j] 这个元素
3. 分割后的路径长度应该 >= 3
我们看一下代码：
```c++
vector<string> t;
bool flag = 0;
string s;
int n;
string add(string a, string b) {
    string c;
    reverse(a.begin(), a.end()); reverse(b.begin(), b.end());
    for (int i = 0, t = 0; i < a.size() || i < b.size() || t; i ++ ) {
        if (i < a.size()) t += a[i] - '0';
        if (i < b.size()) t += b[i] - '0';
        c.push_back(t % 10 + '0');
        t /= 10;
    }
    reverse(c.begin(), c.end());
    return c;
}
void DFS(int idx) {
    if (idx == n) {
        if (t.size() >= 3) flag = 1;    // 分割后的路径长度应该 >= 3
        return;
    }
    if (flag) return;
    for (int j = idx; j < n; j ++ ) {
        string sub = s.substr(idx, j - idx + 1);
        if (sub[0] == '0' && sub.size() > 1) break;     // 合法性剪枝：sub 不能有前缀 0
        if (t.size() >= 2)      // 合法性剪枝：路径的最后一个元素 + 倒数第二个元素 = `当前分割`出来的 s[idx ... j] 这个元素
            if (add(t[t.size() - 1], t[t.size() - 2]) != sub) continue;
        t.push_back(sub);
        DFS(j + 1);
        t.pop_back();
    }
}
bool isAdditiveNumber(string _s) {
    s = _s;
    n = s.size();
    flag = 0;
    DFS(0);
    return flag;
}
```

## 93. 复原 IP 地址
经典的 子集分割型 回溯算法，有难度的地方在于剪枝：

1. sub 如果不是 0 ，那么它的前缀不能有 0
2. 路径长度，即分割后的字符串大小应该为 4 
3. 每一条路径(数字) 都必须 <= 255

代码如下：
```c++
string s;
vector<int> t;
vector<vector<int> > res;
int n;
void DFS(int idx) {
    if (idx == n) { 
        if (t.size() == 4)  // 路径长度，即分割后的字符串大小应该为 4 
            res.push_back(t); 
        return; 
    }
    for (int i = idx; i < n && i < idx + 4; i ++ ) {
        string sub = s.substr(idx, i - idx + 1);
        if (sub.size() > 1 && sub[0] == '0') break;
        int x; stringstream(sub) >> x;      
        if (x > 255) break;         // 每一条路径(数字) 都必须 <= 255
        if (t.size() >= 3 && i + 1 != n) continue;      // 路径长度，即分割后的字符串大小应该为 4 , 如果超过了就剪枝
        t.push_back(x);
        DFS(i + 1);
        t.pop_back();
    }
}
vector<string> restoreIpAddresses(string _s) {
    s = _s; n = s.size();
    DFS(0);
    vector<string> fin;
    for (auto e : res) {
        string tmp;
        for (int i = 0; i < 4; i ++ ) {
            tmp += to_string(e[i]);
            if (i != 3) tmp += ".";
        }
        fin.push_back(tmp);
    }
    return fin;
}
```
## 2698. 求一个整数的惩罚数


子集型回溯法：和 `306. 累加数` 差不多

剪枝方法：我只做了两个剪枝，实际上应该有更多剪枝，不过`我只做我能力范围内的剪枝`

```c++
int flag = 0, len, x, t;
string s;
void DFS(int idx) {
    if (flag) return;
    if (idx == len) { 
        if (t == x) flag = 1; 
        return; 
    }
    for (int i = idx; i < len; i ++ ) {
        string sub = s.substr(idx, i - idx + 1);
        int inc; stringstream(sub) >> inc;
        if (t + inc > x) break;         // t 过大剪枝
        t += inc;
        DFS(i + 1);
        t -= inc;
        if (flag) break;            // flag 标记剪枝
    }
}
int punishmentNumber(int n) {
    int ans = 0;
    for (int i = 1; i <= n; i ++ ) {
        s = to_string(i * i);
        t = 0, flag = 0, x = i, len = s.size();
        DFS(0);
        if (flag) ans += x * x;
    }
    return ans;
}
```

## 2397. 被列覆盖的最多行数
二进制枚举：不用更高级的方法，我现在之学能力范围之内的方法

二进制枚举模板：如果集合一共有 n 个元素，那么：`枚举该集合内的子集`：
```c++
for (int s = 0; s < (1 << n); s ++ ){
    // s 为 `(1 << n) - 1` 这个集合的一个子集，
}

```
```c++
int maximumRows(vector<vector<int>>& matrix, int numSelect) {
    int m = matrix.size(), n = matrix[0].size();
    vector<int> mask(m);
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ )
            mask[i] |= matrix[i][j] << j;

    int ans = 0;
    for (int s = 0; s < (1 << n); s ++ ){
        if (__builtin_popcount(s) != numSelect) continue;
        int cnt_cover = 0;
        for (auto e : mask)
            if ((e & s) == e) cnt_cover ++ ;
        ans = max(ans, cnt_cover);
    }
    return ans;
}
```

## 216. 组合总和 III
方法 1：子集型回溯：由于 `组合 = 子集 + 剪枝` 所以一定不要将 组合问题 看作一个单独的问题，要把它分类到 子集型回溯才算`好习惯`

方法二：DP
我们用 DP 的方法来解决这一问题：

首先定义 f(i, j) 为 前 i 个数中，我们能够凑出 `数字 j` 的所有方案。

那么 f(0, 0) = {{}} 代表凑成 0 有一种方案，那就是啥都不选：{}，其余 f(0, j) = {} 代表没有方案。

千万不能让 f(0, 0) = {} , 否则将无法正确转移！

那么状态转移方程就得到了，我们再进行状态压缩，代码如下：
```c++
vector<vector<int>> combinationSum3(int k, int n) {
    vector<vector<int> > f[n + 1];
    for (int j = 1; j <= n; j++)  f[j] = {};        // 这一步可以没有，因为初始化时，一定会让 f[j] 为 {}
    f[0] = {{}};

    for (int i = 1; i <= 9; i++) 
        for (int j = n; j >= i; j -- ) 
            for (auto vec : f[j - i]) {
                vec.push_back(i);       // 将 j - i 的所有方案加入到 f[j] 中
                f[j].push_back(vec);
            }

    vector<vector<int>> res;
    for (auto& arr : f[n])      // 收集方案数为 k 的方案
        if (arr.size() == k) 
            res.push_back(arr);
    return res;
}
```

## 301. 删除无效的括号
经典的三个手法：
1. 将删除看作是忽略，这样就等价于转换为删除了
2. 利用 子集型回溯来进行 枚举每个字符 -> 选或不选(对应于忽略或不忽略)
3. 如果结束时的路径比答案更优，则替换答案。如果和答案一样优，则加入到答案中去。即与 Dij 中的最短路个数DP 和 最长递增子列的 DP 的思想相同

细节：记得如果有重复字符串，则用 unordered_set 来进行去重

那么我们直接上代码，注意这里的代码没有错误，但是会超时，但是`我们不做我们能力范围之外的剪枝`
```c++
string t;
vector<string> res;
unordered_set<string> S;
int ans = 0, n;
string s;
bool check(string &s) {
    string stk;
    for (int i = 0; i < s.size(); i ++ ) {
        if (isalpha(s[i])) continue;
        else if (s[i] == '(') stk.push_back('(');
        else if (stk.size()) stk.pop_back();
        else return 0;
    }
    return stk.size() == 0;
}
void DFS(int idx) {
    if (idx == n) {
        if (check(t)) {
            if (t.size() > ans)
                ans = t.size(), S.clear(), S.insert(t),
                res.clear(), res.push_back(t);
            else if (t.size() == ans) 
                if (!S.count(t)) res.push_back(t), S.insert(t);
                else return;
                
        }
        return;
    }
    t += s[idx];
    DFS(idx + 1);
    t.pop_back();
    DFS(idx + 1);
    return;
}

vector<string> removeInvalidParentheses(string _s) {
    s = _s;
    t = ""; n = s.size();
    DFS(0);
    return res;
}
```

## 2850. 将石头分散到网格图的最少移动次数
方法 1 ：利用全排列思想来枚举石头 `出发位置 from ` 与 `目标位置 to`

然后我们枚举 from 与 to 进行匹配，即对 from 进行全排列来一一去匹配 to 的对于位置

细节：在进行 next_permutation `之前`一定要进行 sort ，否则将不能完整地遍历所有的 排列情况

```c++
const int INF = 0x3f3f3f3f;
int minimumMoves(vector<vector<int>>& mat) {
    vector<PII> from, to;
    int m = mat.size(), n = mat[0].size();
    for (int i = 0; i < m; i ++ )
        for (int j = 0; j < n; j ++ )
            if (mat[i][j] == 0) 
                to.push_back({i, j});
            else if (mat[i][j] > 1)
                for (int k = 1; k < mat[i][j]; k ++ )
                    from.push_back({i, j});
    sort(from.begin(), from.end());
    int ans = INF;
    do {    
        int t = 0;
        for (int i = 0; i < to.size(); i ++ ) {
            int x1 = to[i].first, y1 = to[i].second, x2 = from[i].first, y2 = from[i].second;
            t += abs(x1 - x2) + abs(y1 - y2);
        }
        ans = min(ans, t);
    } while (next_permutation(from.begin(), from.end()));
    return ans;
}
```

## 985. 查询后的偶数和
很简单的模拟，我们通过计算当前 A[id] 的值是否为 偶数 来决定是否对 sum 产生影响：
```c++
vector<int> sumEvenAfterQueries(vector<int>& A, vector<vector<int>>& queries) {
    int sum = 0;
    for (auto e : A)
        if (e % 2 == 0) sum += e;
    vector<int> res;
    for (auto e : queries) {
        int val = e[0], id = e[1];
        if (A[id] % 2 == 0) sum -= A[id];
        A[id] += val;
        if (A[id] % 2 == 0) sum += A[id];
        res.push_back(sum);
    }
    return res;
}
```

## 473. 火柴拼正方形
用记忆化搜索来剪枝 `排列问题`：

如果不剪枝，那么使用排列的方法回溯该问题，将会导致超时，那么我去记忆 `路径 v`。

问：为什么会重复搜索路径？

答：因为在使用排列的方法时，我们可能会先枚举这个状态： 1....2 (1 和 2 代表`先后枚举的顺序`，而非元素)，然后枚举这个状态：2....1 即我们虽然枚举的顺序不同，但是我们枚举出的 `路径状态` 是一模一样的。所以我们需要`记录下路径状态`，以免重复搜索 `枚举顺序上` 不同，但 `状态上` 相同的路径

我们可以用 state 状态压缩来进行存储状态，也可以用 map 来枚举状态。记住，考场上千万别用 unordered_map 来存 vector, 因为不会相应的语法。

代码如下
```c++
bool flag = 0;
int n;
vector<int> nums;
vector<bool> v;
set<vector<bool> > S;
int side;
void DFS(int idx, int cur) {
    if (idx == 4) {
        flag = 1;
        return;
    }
    if (flag) return;
    if (S.count(v)) return;
    S.insert(v);
    for (int i = idx; i < n; i ++ ) {
        if (v[i] || cur + nums[i] > side) continue;
        v[i] = 1;
        if (cur + nums[i] == side) DFS(idx + 1, 0);
        else DFS(idx, cur + nums[i]);
        v[i] = 0;
        if (flag) return;
    }
}
bool makesquare(vector<int>& _nums) {
    nums = _nums; flag = 0; n = nums.size(); v = vector<bool>(n);
    for (auto e : nums) side += e;
    if (side % 4 != 0) return 0;
    side /= 4;
    DFS(0, 0);
    return flag;
}
```

## 491. 非递减子序列
这题是一道 `子集且不可排列` 的题目，这意味着我们需要去重，但是原来的去重方法：`if (i != idx && nums[i - 1] == nums[i]) continue` 这个方法失效了，因为我们没有对这个数组进行排序！而实际上，我们也不能对原数组进行排序。所以我们`不得不用哈希表`来进行处理重复的情况

代码如下：
```c++
vector<int> t, nums;
vector<vector<int> > res;
int n;
set<vector<int> > S;
void DFS(int idx) {
    if (t.size() >= 2 && !S.count(t)) res.push_back(t), S.insert(t);
    for (int i = idx; i < n; i ++ ) {
        if (t.size() && t.back() > nums[i]) continue;
        t.push_back(nums[i]);
        DFS(i + 1);
        t.pop_back();
    }
}
vector<vector<int>> findSubsequences(vector<int>& _nums) {
    nums = _nums; n = nums.size(); 
    DFS(0);
    return res;
}
```

方法二：剪枝：
由于第一个剪枝方法已经失效，我们使用正确的剪枝方法：`《代码随想录》大佬的剪枝方法`：我们将`同一个节点`(`针对`于搜索树来说)已经用过的数字记录下来，并不再使用它。

搜索树的脑海图应该长这样：即我们一定要将`节点内部的数字`进行`想象`出来

回溯三问：
1. DFS(i) 的含义
2. 选或不选
3. 如何剪枝
```
                                [1, 2, 3, 4, 5]
        [1, 2, 3]       [1, 2, 3, 4]      [1]       [1, 2]      [1, 2, 3]
```
```c++
vector<int> t, nums;
vector<vector<int> > res;
int n;
void DFS(int idx) {
    if (t.size() >= 2) res.push_back(t);
    unordered_set<int> used;
    for (int i = idx; i < n; i ++ ) {
        if (t.size() && t.back() > nums[i]) continue;
        if (used.count(nums[i])) continue;
        used.insert(nums[i]);
        t.push_back(nums[i]);
        DFS(i + 1);
        t.pop_back();
    }
}
vector<vector<int>> findSubsequences(vector<int>& _nums) {
    nums = _nums; n = nums.size(); 
    DFS(0);
    return res;
}
```

## 679. 24 点游戏

我们利用回溯的方法来写此题，这题是需要`模拟的技巧`的：
1. 我们创建一个 `nums[] double 类型数组` 来存储我们需要计算的四个数字
2. 我们从 nums[] 中选取两个数字，然后将他们进行一次计算，然后添加入 next[] 数组中，并将没有选中的数字全部添加到 next[] 中，作为下一次计算的 nums[]
3. 我们在当前回溯节点 nums[] 的大小为 1 时进行收集答案。

`细节`：我们定义 `const double eps = 1e-6;`
1. 我们收集答案的判断不是 nums[0] == 24, 而是 `abs(nums[0] - 24) < eps` 这样能确保不会因为精度损失带来答案错误
2. 我们对于除法中分母判 0 应该是：nums[j] < eps，而不是 nums[j] == 0, 这样能确保不会因为精度损失带来的答案错误

代码如下：
```c++
int flag = 0;
const double eps = 1e-6;
void DFS(vector<double> nums) {
    if (flag) return;
    if (nums.size() == 1) {
        if (abs(nums[0] - 24) < eps) flag = 1;
        return;
    }
    int n = nums.size();
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ ) {
            if (i == j) continue;
            vector<double> next;
            for (int k = 0; k < n; k ++ )
                if (k != i && k != j) next.push_back(nums[k]);
            double x;
            for (int k = 0; k < 4; k ++ ) {
                if (k == 0) x = nums[i] + nums[j];
                else if (k == 1) x = nums[i] - nums[j];
                else if (k == 2) x = nums[i] * nums[j];
                else {
                    if (nums[j] < eps) continue;
                    else x = nums[i] / nums[j];
                }
                next.push_back(x);
                DFS(next);
                next.pop_back();
                if (flag) return;
            }

        }
}
bool judgePoint24(vector<int>& _nums) {
    vector<double> nums(4);
    for (int i = 0; i < 4; i ++ )
        nums[i] = _nums[i];
    DFS(nums);
    return flag;
}  
```

## 钢条切割问题
这是算法导论上的一道题

那么钢条切割问题就是:给定一段长度为 n 英尺的钢条和一个价格表为`Pi(i= 1,2,.. . ,n)`,求切割钢条方案，使得销售收益最大(单位为元)。注意:如果长度为 n 英尺的钢条的价格 Pn 足够大，那么最优解就是不需要切割。

`关键`：
1. 我们的目的`不是去枚举有收益`的长度，而是去枚举每一条`能够切割的长度`，即 1...lenth
2. 我们将 p[] 中没有收益的钢条长度的 收益初始化为 0；

此题为区间 DP 问题，我们定义 DFS(i) 为对长度为 i 的钢条，能够切割出的最佳收益，那么伪代码如下：
```c++
INF = 1e+16 // 无穷大

DFS(n)
    if n == 0
        return 0
    res = p[n]          // 我们的 res 初始化为 没有切割时的 收益
    for i = 1 to n      // 我们去枚举 有利可图的长度 方案：从 p[] 中去选取
        res = max(res, p[i] + DFS(n-i))
    return res
```
翻译递推：
```c++
INF = 1e+16 #无穷大

function(n)
    let arr[0..n] be a new array
    arr[0] = 0
    for j = 1 to n
        &res = arr[j] = p[n]        // 初始化为 没切割时的 收益
        for i = 1 to j
            res = max(res, p[i] + arr[j - i])
        arr[j] = res
    return arr[n]
```
## 2312. 卖木头块
根据钢条切割问题，我们先上代码，在代码中去理解它的含义：

关键：这题和 `鸡蛋掉落` 的`位置无关解法`有异曲同工之妙。都是切完了之后，将它看作是一个新的木板，而切割后的模板与原来木板的位置无关，即将它看作一块新的木板。高楼扔鸡蛋也是从某一楼扔下之后，如果再扔的话，其所处位置直接也不再看作是 `prePosition（上一次的位置） + h`, 而是直接是看作 h。这样`才能满足有重复`的计算，如果每次都当作一个全新的位置，将没有备忘录可以用！！！

即：它是一个区间 DP 问题，但是它应该属于：`位置无关型 区间 DP` 问题。由于它的思想是分区间进行 DP，但是它的转移方程与区间分割后的位置 l, r 根本无关，所以它属于 位置无关型 区间 DP。


枚举技巧：我们第一眼着手就不会做的原因还是因为不会枚举。即不会`暴力枚举`切割位置，才是不会下手的关键。总结：不会`暴力模拟还写个屁的题目`
```c++
vector<vector<LL>> mp, pr;
    
LL DFS(int m, int n) {
    if (mp[m][n] != -1) return mp[m][n];
    LL &res = mp[m][n]; res = pr[m][n];

    for (int i = 1; i <= m / 2; i++) 
        res = max(res, DFS(i, n) + DFS(m - i, n));
    
    for (int i = 1; i <= n / 2; i++) 
        res = max(res, DFS(m, i) + DFS(m, n - i));
    
    return res;
}

long sellingWood(int m, int n, vector<vector<int>>& prices) {
    mp = vector<vector<LL>>(m + 1, vector<LL>(n + 1, -1));
    pr = vector<vector<LL>>(m + 1, vector<LL>(n + 1, 0));

    for (auto& e : prices) {
        pr[e[0]][e[1]] = (LL)e[2];
    }
    return (long)DFS(m, n);
}
```

## 区间 DP 问题总结：
1. 位置有关型：
`Acwing 282. 石子合并`，`LeetCode 1039. 多边形三角剖分的最低得分`，`LeetCode 375. 猜数字大小 II`，`813. 最大平均值和的分组`，`823. 带因子的二叉树`

2. 位置无关型：
`鸡蛋掉落`，`2312. 卖木头块`，`343. 整数拆分`

他们的通用状态方程写法：max(res, DFSi1 + DFSi2 + curi) 其中 DFSi 为在第 i 个位置划分之后的区间，一般`分割之后``有两个区间： DFSi1, DFSi2`，它是否带有位置参数 (l, r) 需要由`该类型是否是位置有关型`来决定。而 cur 则代表利用第 i 个位置分割后所需要的代价或者其他性质。其中 max 函数可以写成 min 或者其他函数。

1. 对于位置有关型，那么可能转移方程为这样：max(res, g(DFS(l, k - 1), DFS(k + 1, r)) + curi)
2. 对于位置无关型，那么可能转移方程为这样：max(res, g(DFS(k), DFS(n - k)) + curi)

其`位置有相关的表达含义`是：我们在当前区间中，划分出了两个区间，如果对于性质来说，位置有关，就要保留相对位置信息，如果位置无关，则无需保留相对位置信息。

## 132. 分割回文串 II
1. 先思考状态定义，
2. 再进行思考我能不能从某个状态进行转移，
3. 最后考虑状态的来源处是否合法。
4. 最最后再思考转移方程

1. f[i] 代表在 1...i 的所有分割方案中，在这所有方案中的最少分割次数。
2. 那么我们处于 i 时，能不能从某个地方转移呢？我们可以从 j 处进行转移。
3. 那么 j 处是否合法呢？如果 j...i 这个子串为回文串，那么他就是合法的转移。
4. 转移方程为：f[i] = f[j] + 1

初始化边界细节：我们让 f[0] = -1 使得能够让一整字符串都是回文串的情况能够正确转移

代码如下：
```c++
int minCut(string s) {
    int n = s.size();
    int f[n + 1]; memset(f, 0x3f, sizeof f); f[1] = 0; f[0] = -1;
    int g[n + 1][n + 1]; memset(g, 0, sizeof g);
    for (int i = 1; i <= n; i ++ ) g[i][i] = 1;
    for (int i = n - 1; i >= 1; i -- )
        for (int j = i + 1; j <= n; j ++ ) {
            if (s[i - 1] == s[j - 1]) {
                if (i + 1 == j) g[i][j] = 1;
                else g[i][j] |= g[i + 1][j - 1];
            }
            else g[i][j] = 0;
        }
    for (int i = 2; i <= n; i ++ )
        for (int j = i - 1; j >= 0; j -- ) {
            if (g[j + 1][i]) {
                f[i] = min(f[i], f[j] + 1);
            }
        }
    return f[n];
}
```

## 2684. 矩阵中移动的最大次数
方法 1 ：动态规划解法：

花了我比较多的时间来想这个判断条件和转移方程。最`耗费我时间`去思考的就是去 `判断当前状态` `是否` 可以`从上一个状态`转移过来，实际上可以用 DFS 和 BFS 来判断，但这样还不如直接使用 BFS 来得简洁。所以

`关键判断`：使用了 f[...][j - 1] > 0 来判别上一个状态是否能从第 1 列到达 j 处。如果 f[...][j - 1] == 0 那么说明第 j - 1 列根本不可能从第 0 列转移过来。那么 j 列就同样不能从 第 0 列 转移过来。

`关键顺序`：我们的遍历顺序是从第 1 列遍历到 n - 1 列。行的遍历顺序无关紧要，`重要的是`将 `列顺序` `放在` 双重循环的 `外层`。

那么其他的什么 只能从小数值转移到大数值就好写了。

代码如下：
```c++
int maxMoves(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    vector<vector<int> > f(m, vector<int>(n));
    int ans = 0;
    for (int j = 1; j < n; j ++ ) {
        for (int i = 0; i < m; i ++ ) {
            if (j - 1 >= 0 && mat[i][j] > mat[i][j - 1] && (f[i][j - 1] != 0 || j == 1)) 
                f[i][j] = max(f[i][j], f[i][j - 1] + 1);
            if (i - 1 >= 0 && j - 1 >= 0 && mat[i][j] > mat[i - 1][j - 1] && (f[i - 1][j - 1] != 0 || j == 1)) 
                f[i][j] = max(f[i][j], f[i - 1][j - 1] + 1);
            if (i + 1 < m && j - 1 >= 0 && mat[i][j] > mat[i + 1][j - 1] && (f[i + 1][j - 1] != 0 || j == 1)) 
                f[i][j] = max(f[i][j], f[i + 1][j - 1] + 1);
            ans = max(ans, f[i][j]);
        }
    }
    return ans;
}
```

方法二：SPFA

也就是 `超级源点 BFS`

还是那句话，只要 BFS 能做，SPFA 一定能做，关键在于你怎么设计 d[][] 数组 和 怎么去更新 d[][] 数组。

直接上代码：
```c++
typedef pair<int, int> PII;
const int dirs[3][2] = { {-1, 1}, {0, 1}, {1, 1} };
int maxMoves(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    int d[m][n]; memset(d, 0, sizeof d);
    int inq[m][n]; memset(inq, 0, sizeof inq);
    queue<PII> q; 
    for (int i = 0; i < m; i ++ ) {
        q.push({i, 0}); inq[0][0] = 1; d[0][0] = 0;
    }
    int ans = 0;
    while (q.size()) {
        for (int i = q.size(); i; i -- ) {
            auto [x, y] = q.front(); q.pop(); inq[x][y] = 0;
            for (int j = 0; j < 3; j ++ ) {
                int nex = x + dirs[j][0], ney = y + dirs[j][1];
                if (nex < 0 || nex >= m || ney < 0 || ney >= n || mat[nex][ney] <= mat[x][y]) continue;
                if (d[nex][ney] < d[x][y] + 1) {
                    d[nex][ney] = d[x][y] + 1;
                    if (!inq[nex][ney])
                        q.push({nex, ney}), inq[nex][ney] = 1;
                    ans = max(ans, d[nex][ney]);
                }
            }
        }
    }
    return ans;
}
```
## 97. 交错字符串
这题一定要想清楚为什么是动态规划，而不是双指针，以及`边界问题`和`转移方程`必须要理清楚。

代码如下：
```c++
bool isInterleave(string s1, string s2, string s3) {
    int m = s1.size(), n = s2.size();
    int f[m + 1][n + 1]; memset(f, 0, sizeof f); f[0][0] = 1;
    if (m + n != s3.size()) return 0;
    for (int i = 1; i <= m; i ++ )
        if (s1[i - 1] == s3[i - 1])
            f[i][0] |= f[i - 1][0];
    for (int j = 1; j <= n; j ++ )
        if (s2[j - 1] == s3[j - 1])
            f[0][j] |= f[0][j - 1];
    for (int i = 1; i <= m; i ++ )
        for (int j = 1; j <= n; j ++ ) {
            if (s1[i - 1] == s3[i + j - 1])
                f[i][j] |= f[i - 1][j];
            if (s2[j - 1] == s3[i + j - 1])
                f[i][j] |= f[i][j - 1];
        }
    return f[m][n];
}
```

## LCR 165. 解密数字
这题就是用 DFS 看着舒服一点，典型的`跳台阶型完全背包`问题。

`技巧积累`：实际上可以直接用字符串来比较：`if (sub <= "25" && sub >= "10")` 而无需转化为整数类型 

代码如下：
```c++
string s;
int memo[33];
int DFS(int i) {
    if (i == -1) return 1;
    if (memo[i] != -1) return memo[i];
    int &res = memo[i]; res = 0;
    res += DFS(i - 1);
    if (i - 1 >= 0) {
        string sub = s.substr(i - 1, 2);
        int subx; stringstream(sub) >> subx;
        if (subx >= 0 && subx <= 25 && sub[0] != '0') res += DFS(i - 2);
    }
    return res;
}
int crackNumber(int x) {
    s = to_string(x);
    memset(memo, -1, sizeof memo);
    return DFS(s.size() - 1);
}
```

## 399. 除法求值
这题非常有趣，实际上可以转化为 `建图 + DFS` 。因为从`数学角度来说`，除法的连乘具有传递性：(a / b) * (b / c) = a / c 所以 `建图之后` ，如果我们要求出 a -> c 的除法，我们只需要进行 DFS 来求出 a -> b 的除法，然后再求出 b -> c 的除法，将他们的边权相乘即可。

细节：
1. 我们的`重载函数`和`输入参数`在 int 与 double 类型之间转换`不会报错`，这样导致了我们很可能将函数参数写为 int，但是我们输入的是 double，就会 `insert` 时，`插入边权`的`结果是 int `，而`不是 double`。这样就会出错。我调试了半天，艹。
2. 需要用 mp 来将 string 与 整数节点进行 一一映射：

```c++
struct Edge{
    int y; double z;
    Edge() {}
    Edge(int y, double z) : y(y), z(z) {}
};
vector<Edge> G[110];
int v[110];
void insert(int x, int y, double z) {   // 注意重载该函数时一定是 double，不然用 int 也不会报语法错，但是结果却和我们想要的不同
    G[x].push_back({y, z});
}
unordered_map<string, int> mp;
int id = 0;
const double eps = 1e-4;
double flag = -1;
void DFS(int x, int tar, double cur) {
    for (auto [y, z] : G[x]) {
        if (y == tar) { flag = cur * z; return; }   // 找到了答案
        if (!v[y]) v[y] = 1, DFS(y, tar, cur * z);      
        if (flag != -1) return;     // 答案性剪枝
    }
}
vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    for (int i = 0; i < equations.size(); i ++ ) {
        string xStr = equations[i][0], yStr = equations[i][1]; double z = values[i];
        if (!mp.count(xStr)) mp[xStr] = id ++ ; if (!mp.count(yStr)) mp[yStr] = id ++ ;
        int x = mp[xStr], y = mp[yStr];
        insert(x, y, z); insert(y, x, 1 / z);
    }
    vector<double> res;
    for (auto e : queries) {
        string xStr = e[0], yStr = e[1];
        if (!mp.count(xStr) || !mp.count(yStr)) { res.push_back(-1); continue; }
        int x = mp[xStr], y = mp[yStr];
        if (x == y) { res.push_back(1); continue; }
        flag = -1; memset(v, 0, sizeof v);
        DFS(x, y, 1);
        res.push_back(flag);
    }
    return res;
}
```

floyd 简化代码复杂性：

注意这里的 floyd 不是为了求最短路，而是为了求两个点的除法举例，所以我们使用的 floyd 的判断条件会很不同。

注意： floyd 还需注意处理一下精度问题，设置一个 eps 来解决。
```c++
vector<vector<double> > G;
void insert(int x, int y, double z){
    G[x][y] = z;
}
unordered_map<string, int> mp;
int id = 0;
const double eps = 1e-4;
vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    G = vector<vector<double> >(50, vector<double>(50, -1));
    for (int i = 0; i < equations.size(); i ++ ) {
        string xStr = equations[i][0], yStr = equations[i][1]; double z = values[i];
        if (!mp.count(xStr)) mp[xStr] = id ++ ; if (!mp.count(yStr)) mp[yStr] = id ++ ;
        int x = mp[xStr], y = mp[yStr];
        insert(x, y, z); insert(y, x, 1 / z);
    }
    for (int k = 0; k < id; k ++ )
        for (int i = 0; i < id; i ++ )
            for (int j = 0; j < id; j ++ ) 
                if (G[i][k] > 0 && G[k][j] > 0)     
                    G[i][j] = G[i][k] * G[k][j];
    vector<double> res;
    for (auto e : queries) {
        string xStr = e[0], yStr = e[1];
        if (!mp.count(xStr) || !mp.count(yStr)) { res.push_back(-1); continue; }
        int x = mp[xStr], y = mp[yStr];
        if (abs(G[x][y] - 1) < eps) G[x][y] = 1;
        res.push_back(G[x][y]);
    }
    return res;
}
```

## LCR 115. 序列重建
这道题就是 2024年 408 数据结构算法题真题，我们需要计算构建的图是否存在唯一的拓扑序列，

判断关键：队列中的元素在任意时刻都不能 > 1。

细节：我们使用 set 进行去重重复的边：

```c++
struct Edge{
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[10010];
int ind[10010];
int n;
void insert(int x, int y){
    G[x].push_back({y}), ind[y] ++ ;
}
int flag = 1;
bool topsort(){
    queue<int> q;
    int cntx = 0;
    for (int i = 1; i <= n; i ++ ) 
        if (ind[i] == 0) q.push(i), cntx ++ ;
    while (q.size()) {
        if (q.size() > 1) return 0;
        for (int i = q.size(); i; i -- ) {
            int x = q.front(); q.pop();
            int cnt = 0;
            for (auto [y] : G[x]) {
                if ( -- ind[y] == 0) q.push(y), cnt ++ ;
            }
        }
    }
    return 1;
}
bool sequenceReconstruction(vector<int>& nums, vector<vector<int>>& sequences) {
    set<PII> edges;
    n = nums.size();
    memset(ind, 0, sizeof ind);
    for (int i = 0; i < sequences.size(); i ++ ) {
        vector<int> vec = sequences[i];
        for (int i = 0; i < vec.size() - 1; i ++ ) {
            int x = vec[i], y = vec[i + 1];
            if (edges.count({x, y})) continue;
            edges.insert({x, y});
            insert(x, y);
        }
    }
    return topsort();
}
```
## 839. 相似字符串组
这题是并查集的应用。

关键：还是那句话，不会暴力还写个屁：我们通过`循环遍历`来`模拟构造`分组。

细节：我们需要通过 mp.count() 进行去重

其他的就很顺其自然了：
```c++
unordered_map<string, int> mp;
int p[310];
int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}
void merge(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return; 
    p[px] = py;
}
bool check(string &s1, string &s2) {        // 检查是否属于同一组
    int n = s1.size();
    int cnt = 0, pos1 = 0, pos2 = 0;
    for (int i = 0; i < n; i ++ )
        if (s1[i] != s2[i]){
            if (cnt == 0) pos1 = i;
            else pos2 = i;
            cnt ++ ;
            if (cnt > 2) return 0;
        }
    if (s1[pos1] == s2[pos2]) return 1;
    return 0;
}
int numSimilarGroups(vector<string>& strs) {
    int id = 0; 
    for (auto e : strs) {
        if (mp.count(e)) continue;      // 去重
        mp[e] = id ++ ;
    }
    for (int i = 0; i < id; i ++ ) p[i] = i;
    for (auto e1 : strs) 
        for (auto e2 : strs) {
            int x = mp[e1], y = mp[e2];
            if (check(e1, e2)) merge(x, y);
        }

    int ans = 0;
    for (int x = 0; x < id; x ++ )
        if (p[x] == x) ans ++ ;
    return ans;
}
```

## 684. 冗余连接
很朴素的暴力循环，`并查集 + set` 这让我更加暴力，我必须学会去暴力的去掉一条边，然后增加一条边来进行暴力。

我天生就是为暴力而生的

```c++
set<vector<int> > S;
int n;
int p[1010];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
void merge(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return;
    p[px] = py;
}
bool check(){
    for (int i = 1; i <= n; i ++ ) p[i] = i;
    for (auto e : S) {
        int x = e[0], y = e[1];
        merge(x, y);
    }
    int cnt = 0;
    for (int i = 1; i <= n; i ++ )
        if (i == p[i]) cnt ++ ;
    if (cnt == 1) return 1;
    return 0;
}
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    n = edges.size();
    for (int i = 0; i < n; i ++ )
        S.insert(edges[i]);
    vector<int> res;
    for (int i = n - 1; i >= 0; i -- ) {
        S.erase(edges[i]);
        if (check()) { res = edges[i]; break;}
        S.insert(edges[i]);
    }
    return res;
}
```

优化：

我们`从第一条边开始`不断合并两个节点，如果第一次找到了一条边，能使得图中出现一个环：也就是说明`该边的两端顶点`都曾经`已`加入过并查集。说明生成了一个环。那么这个边就是我们需要找的最靠后的那个边。因为能组成环的边都在该边之前，而且该边为生成的环中最后一条边

```c++
int n;
int p[1010];
int find(int x) {
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}
void merge(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return;
    p[px] = py;
}
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    n = edges.size();
    for (int i = 1; i <= n; i ++ ) p[i] = i;
    vector<int> res;
    for (int i = 0; i < n; i ++ ) {
        int x = edges[i][0], y = edges[i][1];
        int px = find(x), py = find(y);
        if (px == py) { res = {x, y}; break; }
        merge(x, y);
    }
    return res;
}
```

## 343. 整数拆分
经典的`位置无关型区间 DP`，我们枚举分割点后，由于拆分后的右侧整数和位置无关，就变为了`位置无关型 DP`。

特别注意的是 `813. 最大平均值和的分组` 也是`最难搞定的`就是关于只分一组和分 2 组以上需要分类讨论，不能混为一谈。

`关键性易错点`：我们这里的要求是整数必须被拆分为 `2 个以上`，所以我们的状态设计为 f[i] 为从 [1...i] 拆分后的`两个以上`的整数乘积最大化。所以我们必须分为两步进行求解：
1. 正好拆分为 `两个`，
2. 拆分为 `三个及更多`

如果我们只 拆分为 `三个及更多`，那么拆分为两个的情况就会被忽略，导致结果错误。

代码优化：实际上我们可以通过初始化的工作来合并两个 for，将 正好拆分为 `两个` 与 拆分为 `三个及更多` 这两个 for 循环合并为 1 个，见我之前的提交，我这里就不优化代码了。

代码如下：
```c++
int integerBreak(int n) {
    int f[n + 1]; memset(f, 0, sizeof f); f[0] = 0, f[2] = 1;
    for (int i = 2; i <= n; i ++ ) {
        for (int j = i; j >= 2; j -- )      // 正好拆分为两个
            f[i] = max(f[i], (j - 1) * (i - j + 1));
        for (int j = i; j >= 3; j -- )      // 拆分为 `三个及更多`
            f[i] = max(f[i], f[j - 1] * (i - j + 1));
    }
    return f[n];
}
```

## LCR 138. 有效数字
记录一个暴力写法，通过 '.' 与 'e' 来分割这个数字，然后细节很多，在代码中注释：

我们这里积累一个技巧：

如何`在循环时判断`是否有非法字符：如果 i 指针再一次循环中没有挪动过，说明`当前位置`存在非法字符，让 i 指针不能向后移动。

步骤如下：
1. 将多余的空格去掉
2. 将符号位去掉(如果有的话)
3. 将进行判断数字部分，其中，判断 是否为数字将贯穿始终，所以将 isdigit 放在最后面判断
4. 判断是否有 '.' ，如果有的话判断该处出现的 '.' 是否出现的合法。
5. 判断是否有 'e' 或者 'E' 如果有则 判断该处出现的 'e' 是否合法
6. 最后判断是否出现过数字，或者是否在 'e' 后面出现过数字

```c++
bool validNumber(string s) {
    int i = 0, n = s.size();
    while (s[i] == ' ') i ++ ;      // 1. 将多余的空格去掉
    if (s[i] == '-' || s[i] == '+') i ++ ;  // 2. 将符号位去掉(如果有的话)
    if (!isdigit(s[i]) && s[i] != '.') return 0;    // 合法性判断
    int dot = 0, flag = 0;
    int e = 0;
    while (i < n) {
        int cur = i;                // 判断是否会有非法字符
        if (s[i] == '.') {              // 判断是否有 '.' ，如果有的话判断该处出现的 '.' 是否出现的合法。
            if (e || dot) return 0;     
            dot ++ , i ++ ;
        }
        if (s[i] == 'e' || s[i] == 'E') {       // 判断是否有 'e' 或者 'E' 如果有则 判断该处出现的 'e' 是否合法
            if (!flag) return 0;
            if (e) return 0;
            e ++ , i ++ , flag = 0;     // 我们必须要保证 'e' 后面会出现数字，否则非法，所以置 flag = 0
            if (s[i] == '-' || s[i] == '+') i ++ ;      // 可能在 'e' 后面出现符号位
        }
        if (isdigit(s[i])) { i ++ ; flag = 1; continue; }   // 判断 是否为数字将贯穿始终，所以将 isdigit 放在最后面判断
        if (s[i] == ' ') break;     // 如果出现了空格则代表将结束
        if (i == cur) return 0;     // 如果出现了非法字符，则 i 将不会挪动，返回 0
    }
    while (i < n) {
        if (s[i] == ' ') { i ++ ; continue; }
        else return 0;
    }
    return flag;
}

```
## LCR 164. 破解闯关密码
这题非常的有价值，让我明白了自定义排序几个关键的点：

1. 在 return 进行比较时，绝对不能出现 ==, <= , >= 这些含等号的判断符
2. 如果想要包含 if 语句时，我们一定不能这样写代码：`if(a < b) return a < b; if (a > b) return a > b` 因为这样写是有问题的，如果我们想升序排序，那么我们需要在 a > b 时，需要返回 a < b，即抓住`关键点`：`if 内`一定是 != 符号，绝不能有不等号。
3. 如果想要包含 if 语句时，一定 `是 != 符号，不可以是其他符号` 正确的写代码姿势：`if (a != b) return a < b`。

记忆口诀：如果你要让 a 在 b 的前面，那么 你需要在 a != b 时返回 a < b

而这道题的自定义是比较巧妙的，它是关于两个字符串相加之后进行比较，来进行判断 a 和 b 谁在前在后： a + b < b + a 那么我们让 a 在前， b 在后。这种方法需要积累。

代码如下：
```c++
bool cmp(const string &a, const string &b){
    return a + b < b + a;
}
string crackPassword(vector<int>& password) {
    string res;
    vector<string> t;
    for (auto e : password) 
        t.push_back(to_string(e));
    sort(t.begin(), t.end(), cmp);
    for (auto e : t)
        res += e;
    return res;
}
```


## LCR 165. 解密数字
之前有，请用 `ctrl + f` 进行跳转

## 400. 第 N 位数字
注：如果暴力都不会，还写个屁的题目

这题是一个规律题，我们先看如何暴力破解它：

由于我们需要从 x 的最高位开始枚举，而使用`通常的分解整数法`是从最低位开始分解，那么我们用 to_string() 来暴力分解它是最合适的。

我们从 1 -> x... 一直枚举，并用 i 来记录枚举`分解过后`的`字符整数`已经枚举到第几位了，即以单个字符为枚举单位

```c++
int findNthDigit(int n) {
    int x = 1, i = 0;
    while (1) {
        string t = to_string(x ++ );
        for (auto e : t) {
            i ++ ;
            if (i == n) return e - '0';
        }
    }
    return 0;
}
```



## LCR 185. 统计结果概率
这题需要学习，而不是来作为考题！

定义：f(n, x) 为投掷 n 个色子，点数之和为 x 的概率。那么我们考虑 f(n) 与 f(n - 1) 的关系： f(n, x) 其实`有 6 个从 f(n - 1)`到来的途径，等于 f(n - 1, x - 1)* (1 / 6) + f(n - 1, x - 2) * (1 / 6) +...+ f(n - 1, x - 6) * (1 / 6)。也就是从 f(n - 1) 可以转移到 f(n)，那么为了使得边界条件更好确定，我们选择往后更新策略，即当前的 f(n) 会对 f(n + 1) 产生哪些影响。

细节：当遍历到第 i 个色子时，我们的结果范围为：[i...6 * i]，举例：假如我们有三个色子，那么结果的取值范围为：[3...18]。

对于当前的第 i 个色子时，我们投掷下一个骰子 i + 1, 并通过 i + 1 能取到的值为 [1...6] 来更新 f(i + 1)
```c++
vector<double> statisticsProbability(int num) {
    vector<double> f(7, 1.0 / 6);                   // 初始时，我们不使用第 0 个下标，而是使用第 1...6 个
    for (int i = 1; i < num; i ++ ) {
        vector<double> next((i + 1) * 6 + 1);       // 对于下一层的 i 来说，其结果取值范围为 [i + 1....6 * (i + 1)]，据此创建 next 数组 
        for (int j = i; j < f.size(); j ++ )        // 我们只是用当前第 i 层的 [i...6 * i] 范围内的色子进行更新后续的 f(i + 1)
            for (int k = 1; k <= 6; k ++ )          // 遍历下一个色子能够 掷出的范围: [1...6]，然后更新 next[] 
                next[j + k] += f[j] / 6.0;          // 对于当前的第 i 个色子时，我们投掷下一个骰子 i + 1, 并通过 i + 1 能取到的值为 [1...6] 来更新 f(i + 1)
        f = next;
    }
    vector<double> res;
    for (int i = num; i < f.size(); i ++ )
        res.push_back(f[i]);
    return res;
}
```

## 100255. 成为 K 特殊字符串需要删除的最少字符数
经典的我们要删除最少的字符，转化为保留最多的字符。

1. 先统计每个字符出现的次数到 cnt[] 数组中
2. 将 cnt[] 从小到大排序，并遍历 cnt[] 
3. 对于当前的 cnt[i], 我们考虑这样的一个`出现次数区间：[cnt[i], cnt[i] + k]` ，然后遍历 cnt[] 的所有`字符代表的次数`，次数小于该区间的元素全部删除，而处于该区间的则全部保留，超过该区间的则只保留 cnt[i] + k 个。


代码如下：

```c++
int cnt[26];
int minimumDeletions(string word, int k) {
    memset(cnt, 0, sizeof cnt);
    for (auto e : word)     // 先统计每个字符出现的次数到 cnt[] 数组中
        cnt[e - 'a'] ++ ;
    sort(cnt, cnt + 26);        // 将 cnt[] 从小到大排序，并遍历 cnt[] 
    int remain = 0;             // 经典的我们要删除最少的字符，转化为保留最多的字符
    for (int i = 0; i < 26; i ++ ) {        // 对于当前的 cnt[i], 我们考虑这样的一个`出现次数区间：[cnt[i], cnt[i] + k]`
        int t = 0;
        for (int j = 0; j < 26; j ++ ) {        // 然后遍历 cnt[] 的所有 `字符代表的次数` 
            if (cnt[j] < cnt[i]) continue;      // 次数小于该区间的元素全部删除，
            else if (cnt[j] <= cnt[i] + k) t += cnt[j];     // 而处于该区间的则全部保留
            else if (cnt[j] > cnt[i] + k) t += cnt[i] + k;  // 超过该区间的则只保留 cnt[i] + k 个。
        }   
        remain = max(remain, t);
    }
    return word.size() - remain;
}
```
## 深搜与回溯的区别：

抽象区别：
1. 深搜只需要将所有路径上的节点全部访问过一次即可。`对于访问过`的节点，我们将`不再访问`。
1. 而回溯是需要经过 `每一条，所有` 的路径，也就是说，即使有一个节点已经被访问过了，但是`有不同的路径包含这个同一个节点`，那么这个节点需要被反复访问。


从题目要求上的区别：
1. 回溯一般是用于`构造`， DFS(i) 是`已经构造完`了 [0...i - 1], 我现在`正在`构造 i，我下一步要构造 DFS(i + 1)

## 1334. 阈值距离内邻居最少的城市
这是 floyd 算法的基本应用，我们可以去看高级应用 `3512. 最短距离总和` 这种应用太高级，记住其蕴含的思想更为重要

```c++
int d[110][100];
void insert(int x, int y, int z) {
    d[x][y] = z;
}
int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
    memset(d, 0x3f, sizeof d);
    for (auto e : edges) {
        int x = e[0], y = e[1], z = e[2];
        insert(x, y, z), insert(y, x, z);
    }
    for (int k = 0; k < n; k ++ )
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                if (i != j)
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    int Min = 0x3f3f3f3f, ans;
    for (int i = 0; i < n; i ++ ) {
        int cnt = 0;
        for (int j = 0; j < n; j ++ ) {
            if (d[i][j] <= distanceThreshold) cnt ++ ;
        }
        if (cnt <= Min) 
            Min = cnt, ans = i;
    }
    return ans;
}
```

## 2642. 设计可以求最短路径的图类
这题对加深 floyd 算法的理解非常有帮助，我们来看看我经常忽略的点
1. floyd 算法与其他算法最大的不同的一个点在于：它的邻接矩阵的 d[i][i] 必须被初始化为 0 ！！！

那么我们如何通过添加一条边对 floyd 进行更新呢：`添加一条边的更新操作一定是 四个点，而不是传统的三个点`
```c++
void addEdge(vector<int> e) {           // 一定不要再使用 insert 函数，因为 G[][] 已经是更新过后的图，即它已经相当于
    int x = e[0], y = e[1], z = e[2];   // 是最短的矩阵，而不是存储图的邻接矩阵！！！
    if (z >= G[x][y]) return;           // 我们将 G[][] 看作距离矩阵处理，如果 z 大于 G[x][y] 说明不必要更新
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n;  j++ )
            G[i][j] = min(G[i][j], G[i][x] + z + G[y][j]);  // 这是正确的写法，即 i -> x -> y -> j 一共四个点，而不是初始的三个点的写法！！！
}
```

我们回忆一下 `3512. 最短距离总和`：实际上我们是先`完整构造`了一个`最终状态的图`，然后在加入点的过程中，将最终的 距离矩阵 d[][] 进行更新，这样做是没错的。可是如果是该题：`2642. 设计可以求最短路径的图类` 的话，绝对不能像 `3512. 最短距离总和` 这样加点进行更新，因为题目没有给我们一个完全的 `最终状态图` 。这意味着我们添加一个边后，其他有些节点没加入进来，那么就无法更新那些点对应的距离矩阵上的距离。

## 1793. 好子数组的最大分数


这题可以用单调栈，也可以用 `双指针 + 贪心`。

方法一：单调栈：
单调栈就是算出每个点作为最小值时的`距离自己最近的比自己大`的数的`左下标`和`右下标`。然后需要这两个左右下标形成的区间 [l, r] 包含 k 。那么我们通过单调栈就可求出每个点的 `距离自己最近的比自己大` 的数的 `左下标` 和 `右下标` 然后通过遍历，在 [l, r] 包含 k 时更新答案即可。

代码略，可以`使用一次遍历法`

方法二：中心扩散型双指针 + 贪心：


我们尝试从 i = k, j = k 出发，通过不断移动指针来找到最大矩形。比较 `nums[l - 1]` 和 `nums[r + 1]` 的大小，谁大就移动谁（一样大移动哪个都可以)。

```c++
const int INF = 0x3f3f3f3f;
int maximumScore(vector<int> &nums, int k) {
    int n = nums.size(); 
    int l = k, r = k, ans = 0, minH = INF;
    while (l >= 0 && r < n) {
        minH = min({minH, nums[l], nums[r]});
        ans = max(ans, minH * (r - l + 1));
        if (l == 0 && r == n - 1) break;            // 都到达边界时，更新答案后才能 break。注意 break 的位置
        if (l > 0 && r < n - 1) {
            if (nums[l - 1] < nums[r + 1]) r ++ ;   // 比较 `nums[l - 1]` 和 `nums[r + 1]` 的大小，谁大就移动谁
            else l -- ;                             // 一样大移动哪个都可以
        }
        else if (l == 0) r ++ ;                     // 到达边界时，不可再移动
        else if (r == n - 1) l -- ;
    }
    return ans;
}
```


## 2501. 数组中最长的方波
这题也是寻找 `正确转移源点` 的动态规划，和等差序列类题目差不多，那么我们来看怎么写：

方法 1：动态规划 复杂度 O(n^2)

代码略

方法 2：哈希优化，复杂度 O(n)
```c++
int longestSquareStreak(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    unordered_map<int, int> mp;
    int n = nums.size();
    for (int i = 0; i < n; i ++ ) mp[nums[i]] = i;
    vector<int> f(n, 1);
    int ans = 1;
    for (int i = 1; i < n; i ++ ) {
        int sqt = sqrt(nums[i]);
        if (sqt * sqt == nums[i] && mp.count(sqt)) 
            f[i] = f[mp[sqt]] + 1,
            ans = max(ans, f[i]);
    }
    return ans == 1 ? -1 : ans;
}
```

## 拓扑排序
晴问：https://sunnywhy.com/sfbj/10/6/401

他要求拓扑序列按照`从小到大的顺序`进行排列。

该题最关键的地方非常炸裂，难以分析成功：`必须` 使用小根堆！！！

`关键`：因为他不能按照常理来进行 `分层式有序` 拓扑排序，他一定 `不是分层的` ！！！

举例：一个有着 4 个点，两条边的图如下：
```
1->2, 3->4
```

那么如果是`分层式有序` BFS，那么我们就得到的一定是 [1, 3, 2, 4] 但是正确答案是 [1, 2, 3, 4] 所以我们只能使用小根堆来实现这一操作，`分层式有序 BFS 是错误的做法`！！！

拓展：我们`不应该`去求总共拓扑`路径`的`数量`，这是一个 NP 问题.

代码如下：
```c++
#include <bits/stdc++.h>
using namespace std; 
struct Edge {
    int y;
    Edge() {}
    Edge(int y) : y(y) {}
};
vector<Edge> G[110];
int ind[110];
vector<int> arr;
int n, m;
void insert(int x, int y) {
    G[x].push_back({y}); ind[y] ++ ;
}
void topsort(){
    priority_queue<int, vector<int>, greater<int> > q;
    for (int i = 0; i < n; i ++ )
        if (ind[i] == 0) q.push(i);
    while (q.size()) {
        int x = q.top(); q.pop(), arr.push_back(x);
        for (auto [y] : G[x]) {
            if ( -- ind[y] == 0) q.push(y);
        }
    }
}
int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i ++ ) {
        int x, y; cin >> x >> y;
        insert(x, y);
    }
    topsort();
    for (int i = 0; i < arr.size(); i ++ ) {
        cout << arr[i] ;
        if (i != arr.size() - 1) cout << " ";
    }
    return 0;
}
```

## 求关键活动：
晴问：https://sunnywhy.com/sfbj/10/6/401

这道题的最关键容易犯错的地方：
1. 将 vl 初始化为 INF
2. 不用`逆`拓扑序列对 vl 进行更新
3. vl 进行动态规划时用 ve 进行更新
4. 没有对 一条边进行三重判断：vex = vlx, vey = vly, vex + z = vey

我们为了规避以上两点错误，我们进行一个口诀记忆：
1. 一定不去写 const int INF
2. 一定在`逆`拓扑序列上更新 ve 与 vl
3. ve 一定只用 ve 更新，vl 一定只用 vl 更新
4. 对于关键活动进行三重判断
```c++
#include <bits/stdc++.h>
using namespace std; 
struct Edge {
    int y, z;
    Edge() {}
    Edge(int y, int z) : y(y), z(z) {}
};
vector<Edge> G[110];
int ind[110];
vector<int> arr;
vector<int> ve, vl;
int maxLen;
int n, m;
void insert(int x, int y, int z) {
    G[x].push_back({y, z}); ind[y] ++ ;
}
bool topsort(){
    ve = vector<int>(n, 0);
    queue<int> q;
    for (int i = 0; i < n; i ++ )
        if (ind[i] == 0) q.push(i);
    while (q.size()) {
        int x = q.front(); q.pop(), arr.push_back(x);
        for (auto [y, z] : G[x]) {
            ve[y] = max(ve[y], ve[x] + z);
            if ( -- ind[y] == 0) q.push(y);
        }
    }
    if (arr.size() != n) return 0;
    return 1;
}
void getActive(){
    maxLen = 0;
    for (int i = 0; i < ve.size(); i ++ ) maxLen = max(maxLen, ve[i]);
    vl = vector<int>(n, maxLen);
    for (int i = arr.size() - 1; i >= 0; i -- ) {       // 一定在 `拓扑序列` 上更新 ve 与 vl
        int x = arr[i];
        for (auto [y, z] : G[x]) 
            vl[x] = min(vl[x], vl[y] - z);              // 3. ve 一定只用 ve 更新，vl 一定只用 vl 更新
    }
        
    vector<vector<int>> Act;
    for (int x = 0; x < n; x ++ ) 
        for (auto [y, z] : G[x]) 
            if (ve[x] == vl[x] && ve[y] == vl[y] && ve[x] + z == ve[y])       // 4. 对于关键活动进行三重判断
                Act.push_back({x, y});
    sort(Act.begin(), Act.end());
    for (auto e : Act) {
        cout << e[0] << " " << e[1];
        cout << endl;
    }   
}
int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i ++ ) {
        int x, y, z; cin >> x >> y >> z;
        insert(x, y, z);
    }
    if (topsort()) {
        cout << "Yes" << endl;
        getActive();
    }
    else cout << "No";
    return 0;
}
```

## 求关键路径

就是加上一个回溯算法：回溯部分如下：我们只在关键路径上回溯
```c++
vector<string> res;
string t;
void DFS(int x) {
    if (vl[x] == maxLen) {
        res.push_back(t);
        return;
    }
    for (auto [y, z] : G[x]) {
        if (ve[x] + z == vl[y]) {       // 只在关键路径上回溯
            string record = t;
            t += "->" + to_string(y);
            DFS(y);
            t = record;
        }
    }
}
```
完整代码如下：

```c++
#include <bits/stdc++.h>
using namespace std; 
struct Edge {
    int y, z;
    Edge() {}
    Edge(int y, int z) : y(y), z(z) {}
};
vector<Edge> G[110];
int ind[110];
vector<int> arr;
vector<int> ve, vl;
int maxLen;
int n, m;
void insert(int x, int y, int z) {
    G[x].push_back({y, z}); ind[y] ++ ;
}
bool topsort(){
    ve = vector<int>(n, 0);
    queue<int> q;
    for (int i = 0; i < n; i ++ )
        if (ind[i] == 0) q.push(i);
    while (q.size()) {
        int x = q.front(); q.pop(), arr.push_back(x);
        for (auto [y, z] : G[x]) {
            ve[y] = max(ve[y], ve[x] + z);
            if ( -- ind[y] == 0) q.push(y);
        }
    }
    if (arr.size() != n) return 0;
    return 1;
}
void getActive(){
    maxLen = 0;
    for (int i = 0; i < ve.size(); i ++ ) maxLen = max(maxLen, ve[i]);
    vl = vector<int>(n, maxLen);
    for (int i = arr.size() - 1; i >= 0; i -- ) {
        int x = arr[i];
        for (auto [y, z] : G[x]) 
            vl[x] = min(vl[x], vl[y] - z);
    }
        
    vector<vector<int>> Act;
    for (int x = 0; x < n; x ++ ) 
        for (auto [y, z] : G[x]) 
            if (ve[x] == vl[x] && ve[y] == vl[y] && ve[x] + z == ve[y])
                Act.push_back({x, y});
    sort(Act.begin(), Act.end());
}
vector<string> res;
string t;
void DFS(int x) {
    if (vl[x] == maxLen) {
        res.push_back(t);
        return;
    }
    for (auto [y, z] : G[x]) {
        if (ve[x] + z == vl[y]) {
            string record = t;
            t += "->" + to_string(y);
            DFS(y);
            t = record;
        }
    }
}
int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i ++ ) {
        int x, y, z; cin >> x >> y >> z;
        insert(x, y, z);
    }
    if (topsort()) {
        cout << "Yes" << endl;
        getActive();
        t += to_string(arr[0]);
        DFS(arr[0]);
        t = "";
        sort(res.begin(), res.end());
        for (auto e : res) 
            cout << e << endl;
    }
    else cout << "No";
    return 0;
}
```

## 最小生成树：
关键启发：改进一下 insert 函数，`防止有重边`：
```c++
void insert(int x, int y, int z) {
    G[x][y] = min(G[x][y], z);
}
```
