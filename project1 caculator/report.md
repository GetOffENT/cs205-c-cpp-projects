

# CS205 C/ C++ Programming-Calculator

**Name**: 吴宇贤

**SID**: 12212614

[toc]

## Part 1 - Analysis

### 1. 阅读需求

首先Project文档中提供了最简单的例子：

```
$./calculator 2 + 3
2 + 3 = 5
$./calculator 2 - 3
2 - 3 = -1
$./calculator 2 * 3
2 * 3 = 6
$./calculator 2 / 3
2 / 3 = 0.66666667
```

不难发现，这个项目的核心就是通过命令行参数传入表达式，进行两数的四则运算。

```
./calculator 3.14 / 0
A number cannot be divied by zero.

./calculator a * 2
The input cannot be interpret as numbers!
```

命令行参数以字符串的形式传入，所以要对于参数进行错误检测，如除零、数字不合法等。

```
./calculator 987654321 * 987654321

./calculator 1.0e200 * 1.0e200
```

能够完成大数计算，甚至可以进行科学计数法的运算。

```
./calculator
2 + 3 # the input
2 + 3 = 5 # the output
2 * 3 # input another expression
2 * 3 = 6 # the output
quit # the command to quit
```

然后是这个计算器的拓展功能，不传递参数的时候能够进入计算模式一直读取用户键入的表达式进行计算直到接收quit指令。

那么如何同时完成以上功能呢？我经历了以下几种思路历程：

1. 解析命令行传入的参数后，简单运行程序的+、-、*、/符号计算结果并输出，但是显然，这无法处理大数以及科学计数法。
2. 将参数解析成double类型，但测试之后发现范围过大的数字精度损失比较多，不太符合要求。
3. 四则运算都可以使用高精度完成。

所以我需要实现一个能够支持高精度四则运算的计算器，并且对于小数位数过多的数字，用户可以自行选择保留小数的位数。

### 2. 需求实现

> 下面图片引用自oi-wiki.org

#### 2.1 前期准备

C语言本身并不支持高精度，最好想到的是用一个数组来表示大数，其每一位表示大数的一个数字，但是使用数组并不方便表示小数，对于科学计数法也无计可施，所以我写了一个BigNum的结构体来表示大数，其中array成员存储数字数据，但是这里我用了一个小技巧，下标最小的位置存放的是数字的 **最低位**，即存储反转的字符串，这么做的好处是，数字的长度可能发生变化，但我希望同样权值位始终保持对齐（例如，希望所有的个位都在下标 `[0]`，所有的十位都在下标 `[1]`……）；同时，加、减、乘的运算一般都从个位开始进行，这都给了**反转存储**以充分的理由。利用这一个结构体可以完整的表示大数、小数、正负数，当然也可以表示成科学计数法

```c
typedef struct
{
    int *array;         // 数字, 倒序存储
    int sign;           // 符号
    int length;         // 数字长度
    int decimal_length; // 小数位数
    int exponent;       // 指数部分
} BigNum;
```

但是用户键入的是字符串，并不能直接转化为BigNum对象，并且程序无法直接输出BigNum对象，于是我为BigNum提供了以下功能函数以解析字符串转化为BigNum对象、管理内存、打印BigNum。

- BigNum BigNumConstructor(char *str); （调用函数前str字符串一定通过了错误检测）
  - 返回BigNum型的bigNum
  - 检查首位是否有符号，初始化bigNum.sign
  - 遍历字符串检查是否有关键字符`e`，若有，则含指数，根据`e`后面的数字初始化bigNum.exponent；若没有，则将bigNum.exponent初始化为0
  - 对于`e`前面的数字部分，检查是否有小数点，若有统计小数位数后将数字逆序存至bigNum.array；若没有，则直接将数字逆序存至bigNum.array
- void freeBigNum(BigNum *bigNum)；
  - 释放分配的内存
- void printBigNum(BigNum *bigNum, int scale, int isScientifNotation)；
  - scale：保留的小数位数；isScientifNotation：是否用科学计数法表示
  - 若使用科学计数法输出，则需要先调整这个bigNum为标准的科学计数法，比如把$12e9$调整为$1.2e10$。
    - 这里我写了一个函数void toScientificNotation(BigNum *bigNum); 用于调整bigNum的小数位数和指数部分大小以达到要求
  - 随后分别打印整数部分(若有)、小数点、小数部分和指数部分即可

#### 2.2 错误检测

首先用户键入的运算式需要满足以下格式`<operand1> <operator> <operand2> [-s n]`，其中`[-s n]`表示要保留的`n`位小数，若不输入则保留默认位数，加减乘法保留原始位数，除法默认保留六位小数。

按照以上规则，解析完键入的运算式后，字符串个数应该满足3个或者5个，否则提示`Invalid expression format`。

对于操作数，用int isValidBigNum(char *str)函数检测其是否满足要求。若满足返回1；不满足则返回0。

- 检查符号位
- 小数点只允许出现一次，且不在指数部分
- 'e'或'E'表示指数，只允许出现一次，且不能是第一个字符
- 检查指数后是否有符号位，且指数部分不能出现小数点
- 最后一个字符不能是 'e' 或 'E'

对于操作符op，只允许+、-、*、/、%，用`strchr("+-*/%", op)`检测是否满足要求

若用户输入了保留小数的位数，则检测第四个字符串是否为`-s`或者`-scale`，不满足则提示不合法。

最后对于除零的检测，用int isBigNumZero(const BigNum *num)函数检测大数是否为0，若为0且操作符为`/`或者`%`则提示`A number cannot be divied by zero!`

#### 2.3 高精度加减法

高精度加减法实际上就是竖式加减法：

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\plus.png" style="zoom:70%;" /> <img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\subtraction.png" style="zoom:70%;" />

对于加法而言，就是从最低位开始，将两个加数对应位置上的数码相加，并判断是否达到或超过10。如果达到，那么处理进位：将更高一位的结果上增加1，当前位的结果减少10。

```c
// 从低位开始，逐位相加，carry表示进位
int carry = 0;
for (int i = 0; i < maxLength - 1; i++)
{
    int aDigit = i < a->length ? a->array[i] : 0;
    int bDigit = i < b->length ? b->array[i] : 0;
    int sum = aDigit + bDigit + carry;
    result.array[i] = sum % 10;
    carry = sum / 10;
    result.length++;
}
if (carry)
{
    result.array[result.length] = carry;
    result.length++;
}
```

对于减法而言，从个位起逐位相减，遇到负的情况则向上一位借 1，整体思路与加法完全一致。

```c
// 从低位开始，逐位相减，borrow表示借位
int borrow = 0;
for (int i = 0; i < a->length; i++)
{
	int aDigit = a->array[i];
	int bDigit = i < b->length ? b->array[i] : 0;
	int diff = aDigit - bDigit - borrow;
	if (diff < 0)
	{
		diff += 10;
		borrow = 1;
	}
	else
	{
		borrow = 0;
	}
	result.array[i] = diff;
	if (diff != 0)
		result.length = i + 1;
}
```



在代码的层面来说就是利用数组模拟竖式计算从低位依次计算到高位，而由于BigNum的array成员是倒序存储数字，故而对于整数而言，我们可以直接利用array成员进行运算。但是由于我们的BigNum支持小数运算，实际上的对齐低位可以描述为对齐小数点，小数位数少的用0补齐，于是我写了一个函数用于补全小数位数。

> 若当前大数小数位数大于等于期望位数时则直接return

```c
// 对小数位数进行补全
void padDecimalPlaces(BigNum *bigNum, int newDecimalLength);
```

这样补全小数后，两个数从低位开始又一一对应，只需继续竖式计算即可，结果小数位数即为补全后的小数位数。

而另一层面如果是对于使用科学计数法的大数，我们需要运用到类似于补全小数的原理，先让指数部分相同（方便起见，把指数大的化小），随后再对有效数部分进行加减运算，当然指数相同后，还需要对指数前的数进行小数补全，随后两个使用科学计数法的数字指数部分不变，其前面的数字部分进行加减法即可。

```c
// 调整指数大小 把指数大的调整为和指数小的一致
void adjustExponent(BigNum *bigNum, int newExponent);
```

核心代码：(对于两个操作数BigNum *a, BigNum *b而言)

```c
// 若指数不一致，把指数较大的调整为和指数较小的一致
adjustExponent(a, b->exponent);
adjustExponent(b, a->exponent);
// 若小数点后位数不一致，补全位数少的
padDecimalPlaces(a, b->decimal_length);
padDecimalPlaces(b, a->decimal_length);
```

#### 2.4 高精度乘法

从本质上来说，高精度乘法还是竖式乘法，以下是1337 * 42的竖式计算过程

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\multiplication-long.png" style="zoom:70%;" />

对于$a * b$得到的$result$来说， 忽略进位时，假设$a$和$b$是十进制下的两个数字，它们的每一位分别为$a_j$和$b_k$，其中$j$和$k$表示数字的位数（从0开始计数）。那么，$result$的第$i$位可以近似地表示为所有$j+k=i$时的$a_j*b_k$和。
$$
result[i]=\sum_{j+k=i}a_j*b_k
$$
利用这个特性，我们先算出结果每一位上的数字之后，再从低位到高位统一进位：

```c
// 先计算result的每一位数字
for (int i = 0; i < a->length; i++)
{
    for (int j = 0; j < b->length; j++)
    {
        result.array[i + j] += a->array[i] * b->array[j];
    }
}

// 统一处理进位
for (int i = 0; i < maxLength - 1; i++)
{
    if (result.array[i] >= 10)
    {
        result.array[i + 1] += result.array[i] / 10;
        result.array[i] %= 10;
    }
}
```

对于小数而言，$result$小数位数即为$a$和$b$小数位数之和；对于科学计数法而言，指数部分相加即可。

#### 2.5 高精度除法

高精度除法还是可以利用除法的竖式来解决，不过其中处理有点复杂

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\division.png" style="zoom:70%;" />

竖式长除法实际上可以看作一个逐次减法的过程。例如上图中商数十位的计算可以这样理解：将$45$减去三次$12$后变得小于$12$，不能再减，故此位为$3$，随后计算下一位。

这里我选择把逆序存储的数字倒转成正序存储的数组进行操作，符合竖式思维

```c
// BigNum转数组顺序存储
void BigNumToArray(int *array, BigNum *bigNum)
{
    for (int i = 0; i < bigNum->length; i++)
    {
        array[i] = bigNum->array[bigNum->length - 1 - i];
    }
}
```

这里实现了一个函数 `canSubtract()` 用于判断被除数以下标 `last_dg` 为最低位，是否可以再减去除数而保持非负。此后对于商的每一位，不断调用 `canSubtract()`，并在成立的时候用高精度减法从余数中减去除数，也即模拟了竖式除法的过程。

```c
// 模拟竖式除法 divisor除数、quotient商、remainder余数(初始等于被除数)
for (int i = b->length - 1; i < dividend_length; i++)
{
    // 计算商的第i位
    while (canSubtract(remainder, divisor, i, b->length))
    {
        // 高精度减法
        for (int j = 0; j < b->length; j++)
        {
            remainder[i - j] -= divisor[b->length - 1 - j];
            if (remainder[i - j] < 0)
            {
                remainder[i - j] += 10;
                remainder[i - j - 1] -= 1;
            }
        }
        quotient[i]++;
    }
}
```

随后再根据需要把顺序存储的数组转化为BigNum：

```c
// 顺序存储的数组转BigNum
void ArrayToBigNum(int *array, BigNum *bigNum, int start, int len)
{
    bigNum->length = len - start;
    bigNum->array = (int *)malloc(bigNum->length * sizeof(int));
    for (int i = 0; i < bigNum->length; i++)
    {
        bigNum->array[i] = array[len - 1 - i];
    }
}
```

对于小数除法，通过被除数和除数同时扩大的方式确保除数为整数，随后记录好小数点的位置即可；如果想要保证更高的精度，可以直接给被除数后面加零，在计算完成之后，移动对应位数的小数点即可。

对于科学计数法，指数部分前面的数字正常计算，指数部分相减即可。

> 而这里remainder即为剩下的余数，于是我们可以拓展出取模操作（对整数而言），允许%运算符

#### 2.6 无参运行程序

每次通过命令行参数传入表达式的方法只计算一个表达式有些局限，我们希望能够无参运行程序获得更多模式。

首先第一个模式在计算两数四则运算的基础上，能够不断地从控制台接收并计算，直到用户输入quit指令退出该模式，我把这个模式命名为`Standard Mode`。

其次，个人认为只对于两个数进行四则运算也过于局限，由于我们已经有BigNum结构和计算加减乘除的函数，这里完全利用这些加之栈的结构可以拓展出一个模式来不断接收并计算完整的表达式，这个模式为`Expression Mode`。

对于模式的选择，程序提醒用户输入`1`、`2`或者`quit`分别表示Standard Mode、Expression Mode或者退出程序

如图：

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\ModeSelection.png" style="zoom:70%;" />

##### Standard Mode

使用while循环保证程序一直运行，每次循环迭代都会提示用户输入一个新的表达式，然后使用`fgets`从标准输入读取这个表达式。随后若通过错误检测则进行运算，若没有通过则提示用户重新输入。

##### Expression Mode(拓展)

在这个模式之下，用户可以自由输入表达式，如`(1 + 2) *3 /(4 - 5)`随后程序计算出结果输出。

我选择使用的方法是使用两个栈：一个用于数字（操作数），另一个用于运算符（包括括号）。以下是详细的步骤：

1. **解析和预处理输入**

​	清理和验证输入表达式，例如去除不必要的空格以方便后续读取表达式。

2. **使用栈实现的表达式算法**

- 操作数栈：用于存储操作数。

```c
typedef struct
{
    BigNum items[STACK_SIZE];
    int top;
} BigNumStack;
```

- 运算符栈：用于存储运算符和括号。

```c
typedef struct
{
    char items[STACK_SIZE];
    int top;
} OperatorStack;
```

基本步骤：

- 遍历表达式中的每个标记（数、运算符、括号）

  - 如果是数字，直接压入操作数栈。

  - 如果是运算符，比较其与运算符栈栈顶运算符的优先级：
    - 如果当前运算符优先级更高，将其压入运算符栈。
    - 否则，从运算符栈中弹出运算符，并从操作数栈中弹出相应数量的操作数进行计算，然后将结果压回操作数栈。重复此过程，直到可以安全地将当前运算符压入运算符栈。

  - 如果是左括号`(`，直接压入运算符栈。

  - 如果是右括号`)`，则反复从运算符栈中弹出运算符，并进行计算，直到遇到左括号`(`。左括号`(`只是用来标记括号的开始，一旦完成计算就应该从栈中移除。

- 遍历完所有标记后，如果运算符栈中仍有运算符，继续进行弹出和计算操作，直到运算符栈为空。

3. **处理高精度数值**

- 上述过程中提到的“数字”使用之前定义的`BigNum`结构体来表示和存储，确保高精度。
- 所有的计算操作（加、减、乘、除）都需要使用相应的高精度算法实现。

4. **最终结果**

- 算法完成后，操作数栈中只剩下一个元素，即整个表达式的计算结果。

### 3 项目特色

1. 高精度支持：四则运算都支持高精度整数和小数，除法也不会出现类似于double的精度缺失
2. 错误提示：若用户输入数据有误能够给出准确提示。
3. 科学计数法：支持用户输入用科学计数法表示的数字，以及能够用科学计数法输出。
4. 支持长表达式运算：用户输入长表达式，程序能够按照正确的顺序计算并输出答案。

## Part 2 - Result & Verification

1. 基本功能

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\基本功能.png" />

2. 错误提示

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\错误检测.png" />

3. 高精度和科学计数法

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\1.png" />

4. 保留小数位数

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\scale.png" />

4. Standard Mode

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\StandardMode.png" />

5. Expression Mode

<img src="\\wsl.localhost\Ubuntu-20.04\home\wyx\cs203\projects\project1 caculator\image\ExpressionMode.png" />

## Part 3 - Difficulties & Solutions

### 怎样表示大数

我是c语言初学者，此前只接触过java，最开始看到project题目的时候所有的思考都是从java的角度出发的，比如写一个BigNum类表示大数，但到c语言这边完全不知道如何下手，于是恶补c语言一些基本的结构和语法后才用结构体解决这个问题。

### 小数大数

最开始BigNum结构体只能够表示整数的大数，在写完整数大数的四则运算之后，怎么样用最小的修改成本能够让现有的结构体表示小数呢？

我发现小数的四则运算和整数完全没有区别，不过是小数点需要做处理，于是我给BigNum结构体加了一个成员decimal_length表示小数的位数，0表示没有小数。

这样即使是想要移动小数点也非常方便，只需要改变decimal_length的位数即可。

### 指数和科学计数法

BigNum能够表示小数之后，如何表示指数和科学计数法呢，于是再次加了一个成员exponent表示指数大小，这样一个正常的大数只需要改变两个成员变量decimal_length和exponent就能够直接达成目的。

### 解析长表达式

这个拓展功能花了很多时间，首先是一个输入一个长表达式，我的程序该如何精确的识别数字和符号呢？最后我选择先处理掉多余的空格，然后遍历字符串一一匹配的方法来解析这个表达式。

其次如何实现这个算法，把上学期的dsaa中的栈知识复习一遍之后，最开始我想的是把中缀表达式转化成后缀表达式然后计算结果，但是我不知道一个栈怎么如和同时存储两种不同类型的变量（大数BigNum、 操作符char），然后我发现我们其实可以选择用两个栈，一个存储大数，一个存储操作符，然后同时对栈进行处理即可。

## Part 4 - Conclusion

1. 通过这次的探索和实践，学到了很多新知识，尤其是关于`BigNum`结构体的使用。这不仅加深了我对C语言中结构体的理解，也让我对如何处理和表示大数有了全新的认识。在构建`BigNum`这一过程中，学会了如何设计一个能够存储和操作大数的数据结构，包括如何在结构体中安排各种字段来存储数值的各个部分、符号、长度、小数位数，以及指数部分来支持科学计数法。

   还学习到了如何通过动态内存分配来给大数分配空间，涉及到对`malloc`的使用，以及如何在操作完成后正确地释放内存来避免内存泄露，而这对于我深入理解C语言中的内存管理机制比较有帮助。

   此外，还掌握了字符串处理的技巧，例如如何去除字符串中的空格、如何从字符串中解析出大数的各个组成部分，以及如何处理科学计数法的表示。学会了使用`atoi`函数将字符串转换为整数，并理解了指针运算在字符串处理中的应用。

   在实现`BigNum`的各种操作过程中，比如加、减、乘、除以及指数的调整，深入理解了基本的算术运算原理，学习到了如何在c语言中，尤其是在处理高精度运算时实现这些原理。

2. 写代码就像建房子，而房子不是一下子就能建起来的，而是一步一步地构建。对于`BigNum`这个结构体来说，我是一步一步地给它添加功能，但实际上我写代码的时候非常痛苦，因为新功能有的地方没办法和旧功能兼容。每当我尝试引入一个新特性时，往往需要重新考虑和调整既有的代码结构和逻辑，以确保新旧功能之间的共存。这个过程中，代码的复杂度逐渐增加，有时候甚至需要对原有的设计进行根本性的重构，这无疑增加了开发的难度和工作量。正如在建筑过程中可能需要重新布线或改管，软件开发也需要不断地迭代和改进，以适应功能的增加和需求的变化。但我们其实可以通过从一开始就构建好框架，尽量避免这种情况的发生，比如如果一开始我就同时考虑到整数、小数以及指数的表示，那么在写代码的过程中无论是哪个部分，我都可以直接考虑到兼容的情况。这个project警示了我在写项目之前要先规划好一个大体的框架，而不是在写代码的过程中不断推翻之前的框架去建立新框架。

## Part 5 - Source Code
