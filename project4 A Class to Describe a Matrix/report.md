

# CS205 C/ C++ A Class to Describe a Matrix

**Name**: 吴宇贤

**SID**: 12212614

[toc]

上交文件：mtrix.hpp、main.cpp、CMakeLists.txt、report.pdf

其中mtrix.hpp实现了矩阵库，main.cpp、CMakeLists.txt主要用于测试

## Part 1 - Analysis

### 1. 阅读需求

>Requirements:
>1. The class should support different data types, like the matrix element can be `unsigned`
>  `char`,`short`, `int`, `float`, `double` , etc. The dimension of the matrix is 2.
>2. Do not use a hard copy if a matrix object is assigned to another. Avoid memory leaks
>  and releasing memory multiple times.
>3. Operation overloading including but not limited to =, +, -, ==, *, etc.
>4. Implement ROI(region of interest) to avoid memeory hard copy. The region of a matrix
>  `Matrix a` can be described by another object `Matrix b` . The two objects, a and b,
>     share the same memory.

提取关键词句： different data types, Do not use a hard copy，Operation overloading，ROI， avoid memeory hard copy。

也就是说，我们要实现一个矩阵类，不能有硬拷贝，设计优雅，其中的计算简单高效，API方便易用。

### 2. 需求实现

#### 2.1 前期准备

矩阵类要求支持多种数据类型，比如``unsigned char ``,`` short ``,`` int ``,`` float`` , ``double ``, etc。首先想到的就是cpp中的模板类，所以脑子里面大概有了方向

```c++
template <typename T>
class Matrix{};
```

这里的``T``就是”万能类“，和Java中的泛型有点类似。

#### 2.2 成员变量

从要实现的功能出发，去考虑Matrix类的成员变量，首先最基础的就是矩阵的行和列

```c++
size_t cols, rows;
```

其次就是要存储的矩阵数据，由于project文档要求拷贝只能是浅拷贝并且不能出现``double free``的情况，在project刚发布的时候我想到的是用一个count变量来计data数据被共享的次数，但是后面的课程中我们学到了`ref_count`（引用计数）机制，而这和我之前的想法实则有些类似，于是我顺理成章地使用了C++中的智能指针``std::shared_ptr``来自动管理动态分配的对象的生命周期。引用计数的核心思想是为每个对象维护一个计数器，该计数器记录了有多少个指针指向该对象。使用这种机制的可以让我们简化内存管理，并帮助防止内存泄漏和悬挂指针问题。

```c++
std::shared_ptr<T[]> data;
```

如果只是最简单的矩阵，那么以上的成员变量以及够用了，那我认为这次的project不算难，但是我发现矩阵在实际应用，尤其是在图像处理和计算机视觉中，矩阵多通道的应用场景要多得多。矩阵多通道通常指的是一个包含多个数据层的矩阵，每个数据层代表图像的一个特定方面。这种矩阵在处理颜色图像时很常见，因为颜色图像通常由多个颜色通道组成，如RGB（红、绿、蓝）三通道。所以我设计矩阵类实现了多通道。

```c++
size_t channels;
size_t step;
```

``channels``表示通道数量；``step``表示对齐后的每行步长（字节），主要是为了后续方便计算。

除了多通道矩阵之外，在图像处理和计算机视觉中，ROI代表“Region of Interest”，即“感兴趣区域”。这是图像中一个被特别标记用于进一步分析或处理的矩形区域。使用ROI可以让我们专注于图像中的特定部分，比如进行特征检测、图像分割或其他分析任务。如果矩阵可以定义一个ROI，就相当于我们选择了矩阵中的一个子矩阵或子区域进行操作，在某些场景下的使用频率反而很高，于是我的矩阵类也实现了ROI。

```c++
size_t roiStartCol, roiStartRow;
size_t roiCols, roiRows;
```

所以总的来说，成员变量如下：

```c++
template <typename T>
class Matrix
{
private:
    std::shared_ptr<T[]> data;       // 智能指针管理动态数组，存储矩阵数据
    size_t cols, rows;               // 列数和行数
    size_t channels;                 // 通道数
    size_t step;                     // 对齐后的每行步长（字节）
    size_t roiStartCol, roiStartRow; // ROI的起始列和起始行
    size_t roiCols, roiRows;         // ROI的列数和行数
};
```



#### 2.3 构造器

对于矩阵类的构造器尽可能考虑多的情况，使得矩阵类更user friendly，在创建和初始化矩阵对象的过程中，尽可能适应不同的使用场景，力求在各种场景下都能迅速初始化矩阵。

##### 2.3.1 默认构造函数

```c++
template <typename T>
Matrix<T>::Matrix()
    : cols(0), rows(0), channels(0), step(0), roiStartCol(0), roiStartRow(0), roiCols(0), roiRows(0), data(nullptr) {}
```

- 用途：构造一个空的矩阵，所有属性初始化为零或空，不分配任何数据。后续有对应函数可以填充数据。

##### 2.3.2 基本构造函数

```c++
template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, size_t channels)
    : cols(cols), rows(rows), channels(channels), roiStartCol(0), roiStartRow(0), roiCols(cols), roiRows(rows)
{
    if (cols == 0 || rows == 0 || channels == 0)
    {
        throw std::invalid_argument("Matrix dimensions and channels must be non-zero.");
    }

    size_t elementSize = sizeof(T);
    size_t pitch = cols * channels * elementSize;
    size_t alignment = 64;                             // 选择64字节对齐
    step = (pitch + alignment - 1) & ~(alignment - 1); // 计算对齐后的步长

    void *ptr = aligned_alloc(alignment, step * rows);
    if (!ptr)
    {
        throw std::bad_alloc();
    }
    data = std::shared_ptr<T[]>(static_cast<T *>(ptr), free); // 确保正确删除

    // 初始化内存为零
    std::memset(ptr, 0, step * rows);
}
```

- 参数
  - `rows`：矩阵的行数。
  - `cols`：矩阵的列数。
  - `channels`：每个元素的通道数，默认为1（在类内部申明中指明了默认值）。
- 用途：创建一个具有指定行数、列数和通道数的矩阵。默认情况下，矩阵为单通道，初始化矩阵数据为零，以避免未定义的行为。
- 细节：执行对齐的内存分配，以确保数据访问的高效性。

##### 2.3.3 ROI 构造函数

```c++
template <typename T>
Matrix<T>::Matrix(const Matrix &base, size_t startRow, size_t startCol, size_t roiRows, size_t roiCols)
    : cols(base.cols), rows(base.rows), channels(base.channels), step(base.step), data(base.data),
      roiStartCol(startCol), roiStartRow(startRow), roiCols(roiCols), roiRows(roiRows)
{
    if (startCol >= cols || startCol + roiCols > cols ||
        startRow >= rows || startRow + roiRows > rows)
    {
        throw std::out_of_range("ROI is out of original matrix's bounds.");
    }
}
```

- 参数
  - `base`：基础矩阵，用于创建ROI视图。
  - `startRow`、`startCol`、`roiRow`、`roiCols`：ROI起始行、ROI起始列、ROI的行数、ROI的列数。
- 用途：基于现有矩阵创建一个新的矩阵视图，该视图仅包含原始矩阵的一个子区域。这种构造器常用于图像和信号处理中，无需复制数据，而是使用原矩阵的数据，也就是共享内存。

##### 2.3.4 拷贝构造函数

```c++
template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : data(other.data), cols(other.cols), rows(other.rows), channels(other.channels),
      step(other.step), roiStartCol(other.roiStartCol), roiStartRow(other.roiStartRow),
      roiCols(other.roiCols), roiRows(other.roiRows) {}
```

- 参数
  - `other`：另一个`Matrix`对象。
- 用途：创建一个新的矩阵，其内容是原矩阵的浅拷贝。由于使用`std::shared_ptr`来管理数据，新的矩阵与原矩阵共享相同的数据，只有在修改时才可能发生复制（写时复制）。但由于这里实现简单矩阵类，并一再强调共享内存，我个人认为如果实现了写时复制那浅拷贝、ROI的意义就没有那么大了，如果真的需要在不改变原矩阵数据的情况下修改数据，我也在后面提供了深拷贝的方法，这也不失为一种优解。

##### 2.3.5 移动构造函数

```c++
template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
    : data(std::move(other.data)), // 接管 data 智能指针
      cols(other.cols), rows(other.rows), channels(other.channels),
      step(other.step), roiStartCol(other.roiStartCol), roiStartRow(other.roiStartRow),
      roiCols(other.roiCols), roiRows(other.roiRows)
{
    // 重置原对象，避免原对象销毁时对数据的影响
    other.cols = 0;
    other.rows = 0;
    other.channels = 0;
    other.step = 0;
    other.roiStartCol = 0;
    other.roiStartRow = 0;
    other.roiCols = 0;
    other.roiRows = 0;
}
```

- 参数
  - `other`：另一个`Matrix`对象，用右值引用表示。
- 用途：采用移动语义来构造矩阵，有效地接管`other`的资源。这种方式用于优化性能，减少不必要的数据复制。

##### 2.3.6 从三维向量构造

```
cpp
复制代码
Matrix(const std::vector<std::vector<std::vector<T>>> &vec);
```

- 参数
  - `vec`：一个三维向量，表示多通道的矩阵数据。
- 用途：根据提供的多维向量数据创建矩阵，这使得从标准容器转换到矩阵变得简单。

##### 2.3.7 从单层向量构造（单通道）

```
cpp
复制代码
Matrix(const std::vector<std::vector<T>> &vec);
```

- 参数
  - `vec`：一个二维向量，表示单通道的矩阵数据。
- 用途：根据提供的二维向量数据创建单通道矩阵。

#### 2.4 运算符重载

##### 2.4.1 赋值运算符

- **拷贝赋值运算符**:

  ```C++
  Matrix<T> &operator=(const Matrix<T> &other);
  ```

  用于将一个矩阵的内容复制到另一个矩阵。（浅拷贝）

- **移动赋值运算符**:

  ```C++
  Matrix<T> &operator=(Matrix<T> &&other) noexcept;
  ```

  用于实现矩阵的移动赋值，接管另一个矩阵的资源。

##### 2.4.2 函数调用运算符:

- 函数调用运算符:

  ```C++
  T &operator()(size_t row, size_t col, size_t channel = 0);
  const T &operator()(size_t row, size_t col, size_t channel = 0) const;
  ```

  提供快捷访问矩阵元素的方式，可以直接使用行列坐标（和通道编号）访问元素，通过 `operator()` 提供一种更快速但不进行边界检查的数据访问方式，这是一种很自然的语法，尤其是在模拟数学对象（如矩阵或张量）时，使用起来很直观，某些情况下由于没有多余的检查性能会更好。

  值得注意的是，这里重写()的内部并没有错误检测，所以除此之外，我还提供了一组at()函数，与重写()的效果类似，但是**`at()`** 方法用于提供边界检查。

  ```c++
  template <typename T>
  T &Matrix<T>::at(size_t row, size_t col, size_t channel)
  {
      if (row >= roiRows || col >= roiCols || channel >= channels)
      {
          throw std::out_of_range("Index out of range");
      }
      size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
      return data[index];
  }
  
  template <typename T>
  const T &Matrix<T>::at(size_t row, size_t col, size_t channel) const
  {
      if (row >= roiRows || col >= roiCols || channel >= channels)
      {
          throw std::out_of_range("Index out of range");
      }
      size_t index = (roiStartRow + row) * step / sizeof(T) + (roiStartCol + col) * channels + channel;
      return data[index];
  }
  ```

##### 2.4.3 加法运算符

- **矩阵与矩阵加法**:

  ```c++
  Matrix<T> operator+(const Matrix<T> &other) const;
  ```

  用于实现两个矩阵的逐元素加法，要求两个矩阵的维度（行数、列数、通道数）必须相同。

- **矩阵与标量加法**:

  ```c++
  Matrix<T> operator+(T scalar) const;
  ```

  将标量值加到矩阵的每一个元素上。

- **复合加法**:

  ```c++
  Matrix<T> &operator+=(const Matrix<T> &other);
  Matrix<T> &operator+=(T scalar);
  ```

  这些运算符将另一个矩阵或标量加到当前矩阵上，并更新当前矩阵的值。

- **加法友元函数**

  ```c++
  template <typename T>
  friend Matrix<T> operator+(T scalar, const Matrix<T> &mat);
  ```

  允许将标量与矩阵的每个元素进行加法运算，其中标量在加号的左侧。这提供了双向加法的灵活性，不论标量是在左侧还是右侧，都可以执行加法操作。

##### 2.4.4 减法运算符

- **矩阵与矩阵减法**:

  ```C++
  Matrix<T> operator-(const Matrix<T> &other) const;
  ```

  实现两个矩阵的逐元素减法。

- **矩阵与标量减法**:

  ```c++
  Matrix<T> operator-(T scalar) const;
  ```

  从矩阵的每一个元素中减去一个标量值。

- **复合减法**:

  ```c++
  Matrix<T> &operator-=(const Matrix<T> &other);
  Matrix<T> &operator-=(T scalar);
  ```

  这些运算符从当前矩阵中减去另一个矩阵或标量，并更新当前矩阵的值。

##### 2.4.5 乘法运算符

- **标量乘法**:

  ```C++
  Matrix<T> operator*(T scalar) const;
  ```

  Matrix<T> operator*(T scalar) const;

- **复合标量乘法**:

  ```C++
  Matrix<T> &operator*=(T scalar);
  ```

  将当前矩阵的每个元素乘以一个标量，并更新矩阵。

- **矩阵乘法**:

  ```C++
  template <typename T>
  Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const
  {
      if (cols != other.rows || channels != other.channels)
      {
          throw std::invalid_argument("Matrix dimensions and channels must be compatible for multiplication.");
      }
  
      // 检查是否应该使用 OpenBLAS
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
      {
          return this->multiply_using_openblas(other);
      }
      else
      {
          Matrix result(rows, other.cols, channels);
          for (size_t c = 0; c < channels; ++c)
          {
              for (size_t i = 0; i < rows; ++i)
              {
                  for (size_t j = 0; j < other.cols; ++j)
                  {
                      T sum = 0;
                      for (size_t k = 0; k < cols; ++k)
                      {
                          sum += (*this)(i, k, c) * other(k, j, c);
                      }
                      result(i, j, c) = sum;
                  }
              }
          }
          return result;
      }
  }
  
  ```

  实现两个矩阵的标准矩阵乘法，这里的实现并非纯暴力，而且调用了OpenBLAS库，但由于OpenBLAS库只支持`float`和`double`两种类型的矩阵乘法，所以我们在这里提前判断类型，如果是`float`或者`double`类型则调用OpenBLAS库的矩阵乘法函数，否则就使用暴力计算。

  而调用OpenBLAS库也没有那么简单，因为我设计的矩阵类兼容了多通道，而OpenBLAS不支持多通道，所以我们如果要调用OpenBLAS库进行矩阵乘法的运算，首先要利用功能函数`extractChannel(size_t channel) const`提取对应的通道后，再进行矩阵乘法的计算，然后将对应的单通道矩阵乘法的计算结果依次写入到多通道矩阵中。

  ```C++
  template <typename T>
  Matrix<T> Matrix<T>::extractChannel(size_t channel) const
  {
      if (channel >= channels)
      {
          throw std::out_of_range("Requested channel exceeds available channels.");
      }
  
      Matrix<T> result(rows, cols, 1); // 创建一个新的单通道矩阵
      for (size_t i = 0; i < rows; ++i)
      {
          for (size_t j = 0; j < cols; ++j)
          {
              result(i, j, 0) = (*this)(i, j, channel); // 复制指定通道的数据
          }
      }
      return result;
  }
  ```

- **乘法友元函数**

  ```c++
  template <typename T>
  friend Matrix<T> operator*(T scalar, const Matrix<T> &mat);
  ```

  允许将标量与矩阵的每个元素进行乘法运算，其中标量在乘号的左侧。这种设计确保了乘法操作的对称性，使得标量可以自由地位于操作的任意一侧。

##### 2.4.6 除法运算符

- **标量除法**

  ```C++
  Matrix<T> operator/(T scalar) const;
  Matrix<T> &operator/=(T scalar);
  ```

  将矩阵的每个元素除以一个标量，并可选择更新矩阵。

##### 2.4.7 比较运算符

- **等价比较**

  ```c++
  bool operator==(const Matrix<T> &other) const;
  ```

  检查两个矩阵是否完全相同。

##### 2.4.8 流操作运算符

```c++
template <typename T> friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &mat);
template <typename T> friend std::istream &operator>>(std::istream &in, Matrix<T> &mat);
```

支持从标准输入输出流中读取或写入矩阵数据。

#### 2.5 实用功能

##### 2.5.1 deepCopy

虽然说project文档要求实现的拷贝是浅拷贝，但我认为在以及实现浅拷贝的基础上完全可以新增一个函数deepCopy()返回深拷贝的Matrix对象以应对不同场景。

```c++
template <typename T>
Matrix<T> Matrix<T>::deepCopy() const
{
    Matrix<T> copy(rows, cols, channels);
    size_t totalElements = rows * step / sizeof(T);
    copy.data = std::shared_ptr<T[]>(new T[totalElements], std::default_delete<T[]>());
    std::copy(this->data.get(), this->data.get() + totalElements, copy.data.get());
    return copy;
}
```

深拷贝本质上就是新建了一个对象，其data指向不同的地址，然后将原来的data数据复制过来。

##### 2.5.2 max和min

找到并返回矩阵中的最大值和最小值。这里用的方法就是遍历一遍矩阵找到最大最小值。

##### 2.5.4 clear

清空矩阵内容，重置所有元素为初始状态。这里的初始状态指的是T类型的初始值。

```c++
template <typename T>
void Matrix<T>::clear()
{
    std::fill(data.get(), data.get() + rows * cols * channels, T{});
}
```

##### 2.5.5 getROI

获取定义的ROI区域作为一个新矩阵，也就是抛弃非ROI区域的数据，只保留ROI区域的数据。

##### 2.5.6 subMatrix

```c++
template <typename T>
Matrix<T> Matrix<T>::subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const
{
    if (startRow + numRows > rows || startCol + numCols > cols)
    {
        throw std::out_of_range("Requested subMatrix exceeds matrix dimensions.");
    }

    Matrix sub(numRows, numCols, channels);
    for (size_t i = 0; i < numRows; ++i)
    {
        for (size_t j = 0; j < numCols; ++j)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                sub(i, j, c) = (*this)(startRow + i, startCol + j, c);
            }
        }
    }
    return sub;
}
```

从矩阵中提取一个子矩阵，子矩阵的区域限定从形参传入。

##### 2.5.7 extractChannel

从多通道矩阵中提取特定通道，返回单通道矩阵。而这个功能函数在类内部也有不错的实用效果。

#### 2.6 静态工厂方法

写完前面的构造器、各种功能还是觉得有哪里不到位。想到了设计模式的静态工厂方法，用于创建对象的实例，常常作为类的静态成员函数实现。使用这些静态函数创建矩阵往往会比构造函数更加直观，其次，使用静态工厂方法，能减少构造函数的数量和复杂性，最重要的是更加user friendly。

##### 2.6.1 zero

创建指定大小的元素全部为零的矩阵。

```c++
template <typename T>
Matrix<T> Matrix<T>::zero(size_t rows, size_t cols, size_t channels)
{
    Matrix<T> mat(rows, cols, channels);
    std::fill(mat.data.get(), mat.data.get() + rows * cols * channels, T(0));
    return mat;
}
```

##### 2.6.2 identity

创建单位矩阵

```c++
template <typename T>
Matrix<T> Matrix<T>::identity(size_t size)
{
    Matrix<T> mat(size, size, 1);
    for (size_t i = 0; i < size; ++i)
    {
        mat(i, i, 0) = 1;
    }
    return mat;
}
```

##### 2.6.3 random

创建指定大小的随机元素矩阵，元素值在指定范围内。

```c++
template <typename T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, size_t channels, T minVal, T maxVal)
{
    Matrix<T> mat(rows, cols, channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minVal, maxVal);

    for (size_t i = 0; i < rows * cols * channels; ++i)
    {
        mat.data[i] = static_cast<T>(dis(gen));
    }
    return mat;
}
```

##### 2.6.4 transpose

返回矩阵的转置

```c++
template <typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T> &mat)
{
    Matrix<T> transposed(mat.cols, mat.rows, mat.channels);
    for (size_t i = 0; i < mat.rows; ++i)
    {
        for (size_t j = 0; j < mat.cols; ++j)
        {
            for (size_t c = 0; c < mat.channels; ++c)
            {
                transposed(j, i, c) = mat(i, j, c);
            }
        }
    }
    return transposed;
}
```

### 3 项目特色

1. **多通道支持**

   此矩阵库支持多通道数据处理，在处理图像或其他多维数据时能发挥较大作用。多通道支持使得库能够直接操作彩色图像或具有多个特征的数据集，无需将数据拆分成单独的层。

2. **内存管理与性能优化**

   通过使用`std::shared_ptr<T[]>`智能指针来管理矩阵数据，不仅简化了内存管理，还减少了内存泄漏的风险。此外，通过对数据进行对齐（使用`step`成员变量），进一步优化了内存访问的性能，在大规模计算上效率占优。

3. **支持ROI功能**

   ROI功能允许用户专注于矩阵的一个子区域进行操作，能应用到很多场景，提供了额外的灵活性，允许更有效率地处理数据。

4. **丰富的运算符支持**

   类重载了多种运算符，包括加法、减法、乘法、除法等，使得数学操作直观且易于实现。这种设计使得代码更易读和易维护，同时保持了操作的直观性。

5. **静态工厂方法**

   提供了如`zero`、`identity`、`random`等静态工厂方法，这些方法使创建特定类型的矩阵变得非常简单和直接，从而增强了代码的清晰度和易用性。

6. **扩展性和可维护性**

   类设计允许容易地添加新功能和修改现有功能，不会对使用该库的客户端代码造成太大影响。例如，可以轻松添加新的数学运算或进一步优化内存管理策略，比如后续添加计算矩阵行列式的函数等等。

## Part 2 - Result & Verification

这里使用[Google Test](https://github.com/google/googletest)进行单元测试，根据Google Test文档一步一步学习单元测试方法，然后对于自己实现的矩阵库进行结果的验证，同时我使用valgrind检测程序有无内存泄漏，达成project文档的要求。

单元测试的代码在main.cpp文件中，安装好valgrind后，在运行的过程中加上valgrind --leak-check=full --track-origins=yes --verbose编译选项即可检测是否有内存泄漏。

### 2.1 构造器测试

对于每种构造器、每种类型（`unsigned`、`char`,`short`, `int`, `float`, `double`）都进行了详细的测试，下面放部分代码，其他详见main.cpp。

```c++
#define CHECK_MATRIX_PARAMS(m, expected_rows, expected_cols, expected_channels, type)      \
    EXPECT_EQ(m.getRows(), expected_rows);                                                 \
    EXPECT_EQ(m.getCols(), expected_cols);                                                 \
    EXPECT_EQ(m.getChannels(), expected_channels);                                         \
    EXPECT_EQ(m.getStep(), (expected_cols * expected_channels * sizeof(type) + 63) & ~63); \
    EXPECT_EQ(m.getROIRows(), expected_rows);                                              \
    EXPECT_EQ(m.getROICols(), expected_cols);                                              \
    EXPECT_EQ(m.getROIStartCol(), 0);                                                      \
    EXPECT_EQ(m.getROIStartRow(), 0);                                                      \
    EXPECT_NE(m.getData(), nullptr)

// 参数化构造函数测试
TEST(ConstructorTest, ParameterizedConstructor)
{
    Matrix<unsigned char> m(10, 15, 3);
    CHECK_MATRIX_PARAMS(m, 10, 15, 3, unsigned char);

    Matrix<short> m2(10, 15, 3);
    CHECK_MATRIX_PARAMS(m2, 10, 15, 3, short);

    Matrix<int> m3(10, 15, 3);
    CHECK_MATRIX_PARAMS(m3, 10, 15, 3, int);

    Matrix<float> m4(10, 15, 3);
    CHECK_MATRIX_PARAMS(m4, 10, 15, 3, float);

    Matrix<double> m5(10, 15, 3);
    CHECK_MATRIX_PARAMS(m5, 10, 15, 3, double);
}

// ROI构造函数测试
TEST(ConstructorTest, ROIConstructor)
{
    Matrix<unsigned char> base(10, 15, 3);       // 创建一个基础矩阵，大小为10x15，3个通道
    Matrix<unsigned char> roi(base, 2, 3, 5, 7); // 创建一个起始于(2, 3)的5x7的ROI区域
    CHECK_ROI_PARAMS(roi, base, 5, 7, 3, 2, 3);

    Matrix<unsigned char> base2(10, 10, 1); // 创建一个基础矩阵，大小为10x10，1个通道
    // 检查构造一个超出基础矩阵范围的ROI是否抛出异常
    EXPECT_THROW(Matrix<unsigned char> roi(base2, 8, 8, 5, 5), std::out_of_range);
}
```

全部测试的结果如下：

![](image\constructor.png)

![](image\memory1.png)

测试通过，内存无泄漏

### 2.2 重载运算符测试

部分代码：

```c++
TEST(MatrixAdditionTest, MatrixPlusMatrix_Int) {
    Matrix<int> m1({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    Matrix<int> m2({{{6, 5}, {4, 3}}, {{2, 1}, {0, -1}}});
    Matrix<int> expected({{{7, 7}, {7, 7}}, {{7, 7}, {7, 7}}});

    auto result = m1 + m2;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixSubtractionTest, MatrixMinusScalar_Int) {
    Matrix<int> m({{{10, 20}, {30, 40}}, {{50, 60}, {70, 80}}});
    int scalar = 5;
    Matrix<int> expected({{{5, 15}, {25, 35}}, {{45, 55}, {65, 75}}});

    auto result = m - scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixMultiplicationTest, MatrixTimesScalar_Int) {
    Matrix<int> m({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    int scalar = 2;
    Matrix<int> expected({{{2, 4}, {6, 8}}, {{10, 12}, {14, 16}}});

    auto result = m * scalar;
    EXPECT_TRUE(compareMatrixData(result, expected));
}
```



![](image\PlusMinus.png)

![](image\MulDiv.png)

![](image\memory2.png)

测试通过，内存无泄漏

### 2.3 矩阵乘法

测试代码：

```c++
TEST(MatrixMultiplication, MultiChannelMultiplication)
{
    // 创建两个简单的2x2两通道矩阵
    Matrix<int> m1({{{1, 2}, {3, 4}},
                    {{5, 6}, {7, 8}}});
    Matrix<int> m2({{{1, 0}, {0, 1}},
                    {{1, 1}, {1, 1}}});

    // 执行矩阵乘法
    Matrix<int> result = m1 * m2;

    // 预期结果，每个通道的乘法都是独立的
    Matrix<int> expected({{{4, 4}, {3, 6}},
                          {{12, 8}, {7, 14}}});
    ASSERT_TRUE(compareMatrixData(result, expected));
}

TEST(MatrixMultiplication, ComplexMultiChannelMultiplication)
{
    // 创建一个3x3三通道矩阵
    Matrix<float> m1({{{1.0f, 0.5f, 0.2f}, {2.0f, 1.0f, 0.5f}, {3.0f, 1.5f, 0.8f}},
                      {{4.0f, 2.0f, 1.0f}, {5.0f, 2.5f, 1.3f}, {6.0f, 3.0f, 1.6f}},
                      {{7.0f, 3.5f, 1.8f}, {8.0f, 4.0f, 2.1f}, {9.0f, 4.5f, 2.4f}}});
    Matrix<float> m2({{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
                      {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
                      {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}});

    // 执行矩阵乘法
    Matrix<float> result = m1 * m2;

    // 预期结果
    Matrix<float> expected({{{6.0f, 2.5f, 1.3f}, {5.0f, 3.0f, 1.3f}, {5.0f, 2.5f, 1.5f}},
                            {{15.0f, 5.5f, 2.9f}, {11.0f, 7.5f, 2.9f}, {11.0f, 5.5f, 3.9f}},
                            {{24.0f, 8.5f, 4.5f}, {17.0f, 12.0f, 4.5f}, {17.0f, 8.5f, 6.3f}}});
    ASSERT_TRUE(compareMatrixData(result, expected));
}
```



![](image\matrixmul.png)

![](image\memory3.png)

## Part 4 - Conclusion

在这个项目中，我通过构建一个矩阵处理类（`Matrix` 类）深入学习了多个编程和数学概念，以及它们在实际应用中的交互方式。这个过程不仅加深了我对 C++ 编程语言的理解，特别是在面向对象设计、模板、智能指针、异常处理和性能优化方面，还涵盖了数学运算的实现，如矩阵的乘法、数据结构转换、内存管理等。

### 1. **面向对象设计**

通过定义一个 `Matrix` 类，实践了面向对象的核心原则，如封装、抽象和多态。`Matrix` 类的设计涉及到构造函数、成员变量和成员函数的合理安排，使得它们能够模拟现实世界中矩阵的行为。

### 2. **模板编程**

利用模板，能做到使 `Matrix` 类适应任何数据类型，从而增加了代码的复用性和灵活性。这在实现函数如 `multiply_using_openblas` 和矩阵的基本操作时尤为重要，因为它们需要根据数据类型（如 `float` 或 `double`）调整行为。

### 3. **智能指针与内存管理**

项目中广泛使用 `std::shared_ptr` 来管理矩阵数据的内存，这不仅防止了内存泄漏，还简化了内存的手动管理。通过智能指针，此项目实现了自动内存管理，这是现代 C++ 开发中的一个重要实践。

### 4. **性能优化**

考虑到矩阵运算可能涉及大量数据，性能优化尤为关键。通过使用对齐内存分配和库函数（如 OpenBLAS），能够显著提高矩阵乘法等操作的性能。此外，我还学习了如何根据矩阵的存储方式和访问模式调整循环的顺序，以利用 CPU 缓存更高效地处理数据。

### 5. **异常处理**

在数据输入和函数执行过程中，我学会了使用异常来处理错误情况。这不仅使得 `Matrix` 类更健壮，也提高了代码的安全性和可维护性。