#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define STACK_SIZE 100

typedef struct
{
    int *array;         // 数字, 倒序存储
    int sign;           // 符号
    int length;         // 数字长度
    int decimal_length; // 小数位数
    int exponent;       // 指数部分的值
} BigNum;

// 检查字符串是否合法
int isValidBigNum(char *str);
// 解析字符串转化成BigNum类型
BigNum BigNumConstructor(char *str);
// 释放内存
void freeBigNum(BigNum *bigNum);
// 打印BigNum类型整数
void printBigNum(BigNum *bigNum, int scale, int isScientifNotation);

BigNum add(BigNum *a, BigNum *b);
// a>b的减法
BigNum subtract(BigNum *a, BigNum *b, int com);

BigNum multiply(BigNum *a, BigNum *b);

void padDecimalPlaces(BigNum *bigNum, int newDecimalLength);
BigNum addBigNum(BigNum *a, BigNum *b);
BigNum subtractBigNum(BigNum *a, BigNum *b);
BigNum multiplyBigNum(BigNum *a, BigNum *b);
void divideBigNum(BigNum *a, BigNum *b, BigNum *c, BigNum *d, int scale);

void calculate(char *operator1, char *operator, char * operator2, int scale);

// 以下函数计算表达式的值

// BigNum 栈
typedef struct
{
    BigNum items[STACK_SIZE];
    int top;
} BigNumStack;

// 运算符栈
typedef struct
{
    char items[STACK_SIZE];
    int top;
} OperatorStack;

void initBigNumStack(BigNumStack *stack);
void initOperatorStack(OperatorStack *stack);
void pushBigNum(BigNumStack *stack, BigNum num);
BigNum popBigNum(BigNumStack *stack);
void pushOperator(OperatorStack *stack, char op);
char popOperator(OperatorStack *stack);
char peekOperator(OperatorStack *stack);
int isOperatorStackEmpty(OperatorStack *stack);
int precedence(char op);
BigNum performOperation(BigNum *a, BigNum *b, char op);
void evaluateExpression(char *expression);
void standardMode();
void removeSpaces(char *str);
void expressionMode();

int main(int argc, char *argv[])
{
    if (argc != 4 && argc != 1 && argc != 6)
    {
        printf("Usage: <operand1> <operator> <operand2>\n");
        printf("Or just run the program without any arguments to choose more mode.\n");
        return 1;
    }

    if (argc == 1)
    {
        char input[100]; // 存储用户输入

        printf("Welcome to smart calculator!\nEnter '1' for standard mode, '2' for expression mode, or 'quit' to exit:\n");
        while (1)
        {
            printf("> ");
            if (!fgets(input, sizeof(input), stdin))
            {
                continue;
            }

            input[strcspn(input, "\n")] = 0;

            if (strcmp(input, "1") == 0)
            {
                // 进入标准模式
                standardMode();
                printf("Main menu!\nType '1' for standard mode, '2' for expression mode, or 'quit' to exit:\n");
            }
            else if (strcmp(input, "2") == 0)
            {
                // 进入表达式模式
                expressionMode();
                printf("Main menu!\nType '1' for standard mode, '2' for expression mode, or 'quit' to exit:\n");
            }
            else if (strcmp(input, "quit") == 0)
            {
                // 退出程序
                break;
            }
            else
            {
                printf("Invalid input. Please enter '1', '2', or 'quit'.\n"
                       "Enter '1' for standard mode, '2' for expression mode, or 'quit' to exit:\n");
            }
        }

        printf("Exiting...\n");
        return 0;
    }
    else
    {
        // 检查操作数和操作符
        if (!isValidBigNum(argv[1]))
        {
            printf("The input cannot be interpreted as numbers! Please enter a valid operand (e.g., -123.45).\n");
            return 1;
        }

        char op = argv[2][0];
        if (strchr("+-*/%", op) == NULL || argv[2][1] != '\0')
        {
            printf("Invalid operator. Only +, -, *, / and %% are allowed.\n");
            return 1;
        }

        if (!isValidBigNum(argv[3]))
        {
            printf("The input cannot be interpreted as numbers! Please enter a valid operand (e.g., -123.45).\n");
            return 1;
        }

        if (argc == 4)
        {
            calculate(argv[1], argv[2], argv[3], -1);
        }
        else if (argc == 6)
        {
            if (!(strcmp(argv[4], "-s") || strcmp(argv[4], "-scale")))
            {
                printf("Invalid format for extension argument. Please enter valid arguments to keep n bits after the decimal: -s(-scale) n.\n");
                return 1;
            }

            int scale;
            sscanf(argv[5], "%d", &scale); // 把argv[5]转换为整形

            calculate(argv[1], argv[2], argv[3], scale);
        }
    }

    return 0;
}

int isValidBigNum(char *str)
{
    int hasDecimalPoint = 0;
    int hasExponent = 0;
    int len = strlen(str);
    int start = 0;

    // 检查符号位
    if (str[0] == '+' || str[0] == '-')
    {
        start = 1;
    }

    for (int i = start; i < len; i++)
    {
        if (str[i] >= '0' && str[i] <= '9')
        {
            continue;
        }
        else if (str[i] == '.' && !hasDecimalPoint && !hasExponent)
        {
            // 小数点只允许出现一次，且不在指数部分
            hasDecimalPoint = 1;
        }
        else if ((str[i] == 'e' || str[i] == 'E') && !hasExponent && i != start)
        {
            // 'e'或'E'表示指数，只允许出现一次，且不能是第一个字符
            hasExponent = 1;
            hasDecimalPoint = 1; // 避免指数部分出现小数点
            // 检查指数后是否有符号位
            if (str[i + 1] == '+' || str[i + 1] == '-')
            {
                i++; // 跳过指数符号位
            }
        }
        else
        {
            // 非法字符
            return 0;
        }
    }

    // 最后一个字符不能是 'e' 或 'E'
    if (len > 0 && (str[len - 1] == 'e' || str[len - 1] == 'E'))
    {
        return 0;
    }

    // 合法
    return 1;
}

BigNum BigNumConstructor(char *str)
{
    BigNum bigNum;
    int strLength = strlen(str);
    int startIndex = 0;

    // 检查是否为负数
    if (str[0] == '-')
    {
        bigNum.sign = -1;
        startIndex = 1; // 跳过负号
    }
    else
    {
        bigNum.sign = 1;
    }

    // 检查是否有指数，并初始化exponent
    int eIndex = 0;
    for (; eIndex < strLength; eIndex++)
    {
        if (str[eIndex] == 'e' || str[eIndex] == 'E')
        {
            break;
        }
    }
    bigNum.exponent = atoi(str + eIndex + 1);
    if (eIndex == strLength)
    {
        bigNum.exponent = 0;
    }

    int isDecimal = 0;
    bigNum.decimal_length = 0;
    // 统计小数位数
    for (int i = startIndex; i < eIndex; i++)
    {
        if (isDecimal)
        {
            bigNum.decimal_length++;
        }
        if (str[i] == '.')
        {
            isDecimal = 1;
        }
    }

    bigNum.length = eIndex - startIndex - isDecimal;
    // 申请并分配内存
    bigNum.array = (int *)malloc(bigNum.length * sizeof(int));

    // 低位在前，从字符串末尾开始转换
    if (isDecimal)
    {
        for (int i = 0; i < eIndex; i++)
        {
            int index = eIndex - 1 - i;
            if (i < bigNum.decimal_length)
            {
                bigNum.array[i] = str[index] - '0';
            }
            else if (i > bigNum.decimal_length)
            {
                bigNum.array[i - 1] = str[index] - '0';
            }
        }
    }
    else
    {
        for (int i = 0; i < bigNum.length; i++)
        {
            bigNum.array[i] = str[eIndex - 1 - i] - '0';
        }
    }

    return bigNum;
}

void freeBigNum(BigNum *bigNum)
{
    free(bigNum->array);
    bigNum->array = NULL;
    bigNum->length = 0;
    bigNum->sign = 0;
    bigNum->decimal_length = 0;
}

void toScientificNotation(BigNum *bigNum)
{
    if (bigNum->decimal_length == bigNum->length - 1) // 只有小数部分
    {
        int index = bigNum->length - 1;
        for (; index >= 0; index--)
        {
            if (bigNum->array[index] != 0) // 从高位开始找到第一个非零数字索引
            {
                break;
            }
        }

        // 用newLength存储科学计数法的数字部分
        int newLength = index + 1;
        int *newArray = (int *)malloc(newLength * sizeof(int));
        for (int i = 0; i <= index; i++)
        {
            newArray[i] = bigNum->array[i];
        }
        free(bigNum->array);
        bigNum->array = newArray;
        bigNum->exponent -= bigNum->length - index - 1;
        bigNum->length = newLength;
        bigNum->decimal_length = bigNum->length - 1;
    }
    else
    {
        bigNum->exponent += bigNum->length - 1 - bigNum->decimal_length;
        bigNum->decimal_length = bigNum->length - 1;
    }
}

void printBigNum(BigNum *bigNum, int scale, int isScientifNotation)
{
    if (bigNum->length == 0)
    {
        printf("0\n");
        return;
    }

    // 指定用科学计数法表示或者指数部分不为0，则调整为合法的科学计数法
    if (isScientifNotation || bigNum->exponent != 0)
    {
        toScientificNotation(bigNum);
    }

    if (bigNum->sign == -1)
        printf("-");

    int startIndex = 0;

    if (scale == -1) // 未指定保留小数位数，舍弃末尾多余的0
    {
        while (bigNum->array[startIndex] == 0 && startIndex < bigNum->decimal_length)
        {
            startIndex++;
        }
    }
    else
    {
        padDecimalPlaces(bigNum, scale);
        int index = bigNum->decimal_length - scale - 1;
        if (bigNum->array[index] >= 5)
        {
            while (index + 1 < bigNum->length && bigNum->array[index + 1] == 9)
            {
                bigNum->array[index + 1] = 0;
                index++;
            }

            if (index + 1 >= bigNum->length)
            {
                printf("1");
            }
            else
            {
                bigNum->array[index + 1]++;
            }
        }
        startIndex = bigNum->decimal_length - scale;
    }

    // 打印整数部分
    for (int i = bigNum->length - 1; i >= bigNum->decimal_length; i--)
    {
        printf("%d", bigNum->array[i]);
    }
    // 打印小于1的数小数点前的0
    if (bigNum->length <= bigNum->decimal_length)
    {
        printf("0");
    }

    // 打印小数点
    if (bigNum->decimal_length > startIndex)
    {
        printf(".");
    }
    // 打印小数部分
    for (int i = bigNum->decimal_length - 1; i >= startIndex; i--)
    {
        printf("%d", bigNum->array[i]);
    }

    // 打印指数部分
    if (bigNum->exponent != 0)
    {
        printf("e%d", bigNum->exponent);
    }

    printf("\n");
}

// 比较两个数字大小
int compare(const BigNum *a, const BigNum *b)
{
    // 检查长度(检查长度前已经对小数进行补全)
    if (a->length > b->length)
        return 1;
    if (a->length < b->length)
        return -1;

    // 长度相同，逐位比较
    for (int i = a->length - 1; i >= 0; i--)
    {
        if (a->array[i] > b->array[i])
            return 1;
        else if (a->array[i] < b->array[i])
            return -1;
    }

    // 相等
    return 0;
}

BigNum add(BigNum *a, BigNum *b)
{
    BigNum result;
    // 最大可能长度
    int maxLength = (a->length > b->length ? a->length : b->length) + 1;
    result.array = (int *)malloc(maxLength * sizeof(int));
    result.length = 0;
    result.decimal_length = a->decimal_length;
    result.exponent = a->exponent;

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

    return result;
}

BigNum subtract(BigNum *a, BigNum *b, int com)
{
    BigNum result;
    result.decimal_length = a->decimal_length;
    result.exponent = a->exponent;
    // 若a < b， 则将a，b位置交换再调用函数
    if (com == -1)
    {
        result = subtract(b, a, 1);
        result.sign = -1;
    }
    else
    {
        result.array = (int *)malloc(a->length * sizeof(int));
        result.length = 0;
        result.sign = 1;

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
    }
    return result;
}

// 对小数位数进行补全
void padDecimalPlaces(BigNum *bigNum, int newDecimalLength)
{
    if (newDecimalLength <= bigNum->decimal_length)
        return; // 不需要补位

    int newLength = bigNum->length + (newDecimalLength - bigNum->decimal_length);
    int *newArray = (int *)malloc(newLength * sizeof(int));

    for (int i = 0; i < newDecimalLength - bigNum->decimal_length; i++)
    {
        newArray[i] = 0;
    }
    for (int i = 0, j = newDecimalLength - bigNum->decimal_length; i < bigNum->length; i++, j++)
    {
        newArray[j] = bigNum->array[i];
    }

    free(bigNum->array);
    bigNum->array = newArray;
    bigNum->decimal_length = newDecimalLength;
    bigNum->length = newLength;
}
// 调整指数大小 把指数大的调整为和指数小的一致
void adjustExponent(BigNum *bigNum, int newExponent)
{
    if (newExponent >= bigNum->exponent)
        return; // 不需要调整

    int exponentChange = bigNum->exponent - newExponent;
    if (exponentChange > bigNum->decimal_length)
    {
        int newLength = bigNum->length + (exponentChange - bigNum->decimal_length);
        int *newArray = (int *)malloc(newLength * sizeof(int));

        for (int i = 0; i < exponentChange - bigNum->decimal_length; i++)
        {
            newArray[i] = 0;
        }
        for (int i = 0, j = exponentChange - bigNum->decimal_length; i < bigNum->length; i++, j++)
        {
            newArray[j] = bigNum->array[i];
        }

        free(bigNum->array);
        bigNum->array = newArray;
        bigNum->decimal_length = 0;
        bigNum->length = newLength;
        bigNum->exponent = newExponent;
    }
    else
    {
        bigNum->decimal_length -= exponentChange;
        bigNum->exponent = newExponent;
    }
}

BigNum addBigNum(BigNum *a, BigNum *b)
{
    if (a->exponent != 0 || b->exponent != 0)
    {
        toScientificNotation(a);
        toScientificNotation(b);
    }

    // 若指数不一致，把指数较大的调整为和指数较小的一致
    adjustExponent(a, b->exponent);
    adjustExponent(b, a->exponent);
    // 若小数点后位数不一致，补全位数少的
    padDecimalPlaces(a, b->decimal_length);
    padDecimalPlaces(b, a->decimal_length);

    BigNum result;
    if (a->sign == 1 && b->sign == 1) // a > 0, b > 0
    {
        result = add(a, b);
        result.sign = 1;
    }
    else if (a->sign == -1 && b->sign == -1) // a < 0, b < 0
    {
        result = add(a, b);
        result.sign = -1;
    }
    else if (a->sign == 1) // a > 0, b < 0
    {
        result = subtract(a, b, compare(a, b));
    }
    else
    { // a < 0, b > 0
        result = subtract(b, a, compare(b, a));
    }

    return result;
}

BigNum subtractBigNum(BigNum *a, BigNum *b)
{
    if (a->exponent != 0 || b->exponent != 0)
    {
        toScientificNotation(a);
        toScientificNotation(b);
    }

    // 若指数不一致，把指数较大的调整为和指数较小的一致
    adjustExponent(a, b->exponent);
    adjustExponent(b, a->exponent);

    // 若小数点后位数不一致，补全位数少的
    padDecimalPlaces(a, b->decimal_length);
    padDecimalPlaces(b, a->decimal_length);

    BigNum result;
    if (a->sign == 1 && b->sign == 1) // a > 0, b > 0
    {
        result = subtract(a, b, compare(a, b));
    }
    else if (a->sign == -1 && b->sign == -1) // a < 0, b < 0
    {
        result = subtract(b, a, compare(b, a));
    }
    else if (a->sign == 1) // a > 0, b < 0
    {
        result = add(a, b);
        result.sign = 1;
    }
    else
    { // a < 0, b > 0
        result = add(a, b);
        result.sign = -1;
    }
    return result;
}

BigNum multiplyBigNum(BigNum *a, BigNum *b)
{
    BigNum result;
    int maxLength = a->length + b->length;
    result.array = (int *)malloc(maxLength * sizeof(int));
    memset(result.array, 0, maxLength * sizeof(int));
    result.length = 0;
    result.sign = a->sign * b->sign;
    result.decimal_length = a->decimal_length + b->decimal_length;
    result.exponent = a->exponent + b->exponent;

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

    for (int i = maxLength - 1; i >= 0; i--)
    {
        if (result.array[i] != 0)
        {
            result.length = i + 1;
            break;
        }
    }

    return result;
}

// BigNum转数组顺序存储
void BigNumToArray(int *array, BigNum *bigNum)
{
    for (int i = 0; i < bigNum->length; i++)
    {
        array[i] = bigNum->array[bigNum->length - 1 - i];
    }
}

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

// last_dg表示最低位, len 表示除数的长度
bool canSubtract(int *a, int *b, int last_dg, int len)
{
    // 被除数剩余的部分比除数长
    if (last_dg - len >= 0 && a[last_dg - len] != 0)
        return true;
    // 从高位到低位，逐位比较
    for (int i = last_dg - len + 1, j = 0; j < len; i++, j++)
    {
        if (a[i] > b[j])
            return true;
        if (a[i] < b[j])
            return false;
    }
    // 相等
    return true;
}

// 判断 BigNum 是否为0
int isBigNumZero(const BigNum *num)
{
    // 如果长度为0，则认为是0
    if (num->length == 0)
        return 1;

    // 检查每一位是否都是0
    for (int i = 0; i < num->length; i++)
    {
        if (num->array[i] != 0)
            return 0; // 发现非0位，不是0
    }

    // 所有位都是0
    return 1;
}

// 计算a/b ，c表示商，d表示余数
void divideBigNum(BigNum *a, BigNum *b, BigNum *c, BigNum *d, int scale)
{
    // 计算到保留位数的后一位
    int dividend_length = scale - (a->decimal_length - b->decimal_length) + 1 + a->length;
    int *dividend = (int *)malloc(dividend_length * sizeof(int));
    memset(dividend, 0, dividend_length * sizeof(int));
    BigNumToArray(dividend, a);

    int *divisor = (int *)malloc(b->length * sizeof(int));
    BigNumToArray(divisor, b);

    int *quotient = (int *)malloc(dividend_length * sizeof(int));
    memset(quotient, 0, dividend_length * sizeof(int));

    int *remainder = (int *)malloc(dividend_length * sizeof(int));
    for (int i = 0; i < a->length; i++)
    {
        remainder[i] = dividend[i];
    }

    // 模拟竖式除法
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

    int start = 0;
    for (; start < dividend_length - scale - 1; start++)
    {
        if (quotient[start] != 0)
        {
            break;
        }
    }
    c->decimal_length = scale + 1;
    c->sign = a->sign * b->sign;
    c->exponent = a->exponent - b->exponent;
    ArrayToBigNum(quotient, c, start, dividend_length);

    start = 0;
    while (remainder[start] == 0)
    {
        start++;
    }
    d->decimal_length = 0;
    d->sign = 1;
    d->exponent = 0;
    ArrayToBigNum(remainder, d, start, dividend_length);
}
int hasENotation(char *operator)
{
    for (size_t i = 0; i < strlen(operator); i++)
    {
        if (operator[i] == 'e' || operator[i] == 'E')
        {
            return 1;
        }
    }
    return 0;
}

void calculate(char *operator1, char *operator, char * operator2, int scale)
{
    BigNum a = BigNumConstructor(operator1);
    BigNum b = BigNumConstructor(operator2);
    BigNum result;
    BigNum temp;

    if (*operator== '/' && isBigNumZero(&b))
    {
        printf("A number cannot be divied by zero!\n");
        freeBigNum(&a);
        freeBigNum(&b);
        return;
    }

    printf("%s %s %s = ", operator1, operator, operator2);

    switch (*operator)
    {
    case '+':
        result = addBigNum(&a, &b);
        break;
    case '-':
        result = subtractBigNum(&a, &b);
        break;
    case '*':
        result = multiplyBigNum(&a, &b);
        break;
    case '/':
        // 除法默认保留六位小数
        if (scale == -1)
        {
            scale = 6;
        }
        divideBigNum(&a, &b, &result, &temp, scale);
        freeBigNum(&temp);
        break;
    case '%':
        divideBigNum(&a, &b, &temp, &result, -1);
        freeBigNum(&temp);
        break;
    default:
        break;
    }

    printBigNum(&result, scale, hasENotation(operator1) || hasENotation(operator2));

    freeBigNum(&a);
    freeBigNum(&b);
    freeBigNum(&result);
}

// 初始化 BigNum栈
void initBigNumStack(BigNumStack *stack)
{
    stack->top = -1;
}

// 初始化运算符栈
void initOperatorStack(OperatorStack *stack)
{
    stack->top = -1;
}

// BigNum入栈操作
void pushBigNum(BigNumStack *stack, BigNum num)
{
    if (stack->top < STACK_SIZE - 1)
    {
        stack->items[++stack->top] = num;
    }
    else
    {
        printf("BigNum Stack overflow\n");
    }
}
// BigNum弹栈操作
BigNum popBigNum(BigNumStack *stack)
{
    if (stack->top >= 0)
    {
        return stack->items[stack->top--];
    }
    else
    {
        printf("BigNum Stack underflow\n");
        return (BigNum){NULL, 0, 0}; // 返回一个空的BigNum作为错误指示
    }
}

// 运算符入栈操作
void pushOperator(OperatorStack *stack, char op)
{
    if (stack->top < STACK_SIZE - 1)
    {
        stack->items[++stack->top] = op;
    }
    else
    {
        printf("Operator Stack overflow\n");
    }
}
// 运算符弹栈操作
char popOperator(OperatorStack *stack)
{
    if (stack->top >= 0)
    {
        return stack->items[stack->top--];
    }
    else
    {
        printf("Operator Stack underflow\n");
        return '\0';
    }
}
// 访问运算符栈栈顶元素
char peekOperator(OperatorStack *stack)
{
    if (stack->top >= 0)
    {
        return stack->items[stack->top];
    }
    else
    {
        return '\0';
    }
}
// 判断运算符栈是否为空
int isOperatorStackEmpty(OperatorStack *stack)
{
    return stack->top == -1;
}
// 返回运算符优先级，数字越大优先级越高
int precedence(char op)
{
    switch (op)
    {
    case '+':
    case '-':
        return 1;
    case '*':
    case '/':
        return 2;
    default:
        return -1;
    }
}
// 对BigNum进行运算(和calculate()类似，参数不同)
BigNum performOperation(BigNum *a, BigNum *b, char op)
{
    BigNum result;
    BigNum temp;

    switch (op)
    {
    case '+':
        result = addBigNum(a, b);
        break;
    case '-':
        result = subtractBigNum(a, b);
        break;
    case '*':
        result = multiplyBigNum(a, b);
        break;
    case '/':
        divideBigNum(a, b, &result, &temp, 6);
        freeBigNum(&temp);
        break;
    default:
        printf("Unsupported operation: %c\n", op);
        break;
    }

    freeBigNum(a);
    freeBigNum(b);
    return result;
}

// 去除表达式中的空格
void removeSpaces(char *str)
{
    int i = 0, j = 0;
    while (str[i])
    {
        if (str[i] != ' ')
        {
            str[j++] = str[i]; // 复制非空格字符
        }
        i++;
    }
    str[j] = '\0'; // 添加字符串结束符
}
// 处理表达式
void evaluateExpression(char *expression)
{
    char initExpression[strlen(expression)];
    strcpy(initExpression, expression);
    removeSpaces(expression);
    BigNumStack numbers;     // 存放BigNum的栈
    OperatorStack operators; // 存放运算符的栈
    initBigNumStack(&numbers);
    initOperatorStack(&operators);

    for (int i = 0; i < strlen(expression); ++i)
    {
        if (isdigit(expression[i]))
        {
            // 处理数字
            char numStr[64]; // 存储数字字符串
            int len = 0;
            while (i < strlen(expression) && (isdigit(expression[i]) || expression[i] == '.' || expression[i] == 'e' || expression[i] == 'E'))
            {
                numStr[len++] = expression[i++];
            }
            numStr[len] = '\0';
            BigNum num = BigNumConstructor(numStr);
            pushBigNum(&numbers, num);
            --i;
        }
        else if (expression[i] == '(')
        {
            // 遇到左括号直接压栈
            pushOperator(&operators, expression[i]);
        }
        else if (expression[i] == ')')
        {
            // 遇到右括号，弹出运算符并计算，直到遇到左括号
            while (!isOperatorStackEmpty(&operators) && peekOperator(&operators) != '(')
            {
                BigNum b = popBigNum(&numbers);
                BigNum a = popBigNum(&numbers);
                char op = popOperator(&operators);
                if (op == '/' && isBigNumZero(&b))
                {
                    printf("A number cannot be divied by zero!\n");
                    freeBigNum(&a);
                    freeBigNum(&b);
                    return;
                }
                BigNum result = performOperation(&a, &b, op);
                pushBigNum(&numbers, result);
                freeBigNum(&a);
                freeBigNum(&b);
            }
            popOperator(&operators); // 弹出左括号
        }
        else if (strchr("+-*/", expression[i]) != NULL)
        {
            // 处理运算符，考虑优先级
            while (!isOperatorStackEmpty(&operators) && precedence(peekOperator(&operators)) >= precedence(expression[i]))
            {
                BigNum b = popBigNum(&numbers);
                BigNum a = popBigNum(&numbers);
                char op = popOperator(&operators);
                if (op == '/' && isBigNumZero(&b))
                {
                    printf("A number cannot be divied by zero!\n");
                    freeBigNum(&a);
                    freeBigNum(&b);
                    return;
                }
                BigNum result = performOperation(&a, &b, op);
                pushBigNum(&numbers, result);
                freeBigNum(&a);
                freeBigNum(&b);
            }
            pushOperator(&operators, expression[i]);
        }
    }

    // 表达式遍历完成后，处理剩余的运算符
    while (!isOperatorStackEmpty(&operators))
    {
        BigNum b = popBigNum(&numbers);
        BigNum a = popBigNum(&numbers);
        char op = popOperator(&operators);
        if (op == '/' && isBigNumZero(&b))
        {
            printf("A number cannot be divied by zero!\n");
            freeBigNum(&a);
            freeBigNum(&b);
            return;
        }
        BigNum result = performOperation(&a, &b, op);
        pushBigNum(&numbers, result);
        freeBigNum(&a);
        freeBigNum(&b);
    }

    // numbers 栈顶的BigNum元素即为最终结果
    BigNum finalResult = popBigNum(&numbers);
    printf("%s = ", initExpression);
    printBigNum(&finalResult, finalResult.decimal_length >= 6 ? 6 : -1, 0);
    freeBigNum(&finalResult);
}
// 标准模式
void standardMode()
{
    printf("Standard Mode!\n");
    printf("Enter expressions in the format <operand1> <operator> <operand2>.\n");
    printf("Type 'quit' to back to main menu.\n");

    char input[256];
    while (1)
    {
        printf("<Standard Mode> ");
        if (!fgets(input, sizeof(input), stdin))
        {
            break;
        }

        // 移除末尾的换行符
        input[strcspn(input, "\n")] = 0;

        // 检查是否退出
        if (strcmp(input, "quit") == 0)
        {
            break; // 退出循环
        }

        // 解析输入的表达式
        char operand1[128], operand2[128], op[2], str[10], extra[2];
        int scale = -1; // 默认的scale值
        int numParsed = sscanf(input, "%s %s %s %s %d %s", operand1, op, operand2, str, &scale, extra);
        if (numParsed == 3)
        {
            if (!isValidBigNum(operand1) || !isValidBigNum(operand2) || (strchr("+-*/%", op[0]) == NULL || op[1] != '\0'))
            {
                printf("Invalid expression format.\nPlease follow the format <operand1> <operator> <operand2>.\n");
                continue;
            }
            calculate(operand1, op, operand2, scale);
        }
        else if (numParsed == 5)
        {
            if (!isValidBigNum(operand1) || !isValidBigNum(operand2) || (strchr("+-*/%", op[0]) == NULL || op[1] != '\0'))
            {
                printf("Invalid expression format.\nPlease follow the format <operand1> <operator> <operand2>.\n");
                continue;
            }
            if (!(strcmp(str, "-s") || strcmp(str, "-scale")))
            {
                printf("Invalid format for extension argument. Please enter valid arguments to keep n bits after the decimal: -s(-scale) n.\n");
                return;
            }
            calculate(operand1, op, operand2, scale);
        }
        else
        {
            printf("Invalid input format. Please follow the format <operand1> <operator> <operand2>.\n");
        }
    }
}
// 表达式模式
void expressionMode()
{
    printf("Expression Mode!\n");
    printf("Enter your expression (type 'quit' to back to main menu):\n");

    char expression[1024]; // 表达式最大长度1024
    while (1)
    {
        printf("<Expression Mode> ");
        if (!fgets(expression, sizeof(expression), stdin))
        {
            break;
        }

        expression[strcspn(expression, "\n")] = 0;

        if (strcmp(expression, "quit") == 0)
        {
            break;
        }
        // 函数处理并计算表达式
        evaluateExpression(expression);
    }
}