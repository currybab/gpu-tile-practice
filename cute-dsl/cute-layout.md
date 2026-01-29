# Cute Layout

CuTe에서 Layout은 “좌표 공간(coordinate space) → 인덱스 공간(index space)”으로 가는 매핑 함수이다. 
즉, 다차원 배열 접근을 “형태(Shape)”와 “stride(Stride)”로 추상화해서, 메모리 배치(row-major/column-major 등)가 바뀌어도 알고리즘 코드를 거의 그대로 유지하게 만드는 게 핵심이다.

여기서 중요한 관점은,
- Layout을 "데이터를 어떻게 저장했는지" 설명하는 메타데이터로만 보지 말고,
- "좌표를 넣으면 어떤 선형 인덱스가 나오는지" 계산하는 함수로 보면 이후(algebra)가 훨씬 자연스럽게 이어진다.

## Fundamental Types and Concepts

### Integers

CuTe는 정적 정수와 동적 정수를 동일하게 처리하려고 시도한다. 모든 동적 정수는 정적 정수로 대체될 수 있으며 그 반대도 가능하다고 한다. 
CuTe에서 "정수"라고 말할 때는 거의 항상 정적 정수 또는 동적 정수를 의미한다.
Python DSL에서는 보통, 정적 상수는 `cutlass.Constexpr[int]` 타입으로 받은 값, 동적 정수는 `cutlass.Int32` 같은 런타임 타입으로 받은 값이다.

### Tuple & IntTuple

튜플은 0개 이상의 요소로 이루어진 유한하고 순서가 있는 리스트이다.

CuTe는 IntTuple 개념을 정수 또는 IntTuple의 튜플로 정의한다. 이 정의는 재귀적이여서 중첩 튜플이 가능하다.
Python DSL에서는 그냥 정수 또는 정수로 이루어진 튜플로 표현한다.

#### 정의된 연산

- `cute.rank(IntTuple)`: `IntTuple`의 요소 개수. 단일 정수는 랭크가 1이며, 튜플은 랭크가 tuple_size임.
- `cute.depth(IntTuple)`: 계층적 `IntTuple`의 개수이다. 단일 정수는 깊이가 0이고, 정수 튜플은 깊이가 1이며, 정수 튜플을 포함하는 튜플은 깊이가 2인 식이다.
- `cute.size(IntTuple)`: `IntTuple`의 모든 요소의 곱
- `cute.get(IntTuple, I)`: `IntTuple`의 `I`번째 요소

```python 
t = ((1,2), (3,4))
cute.printf("t.rank={}, t.depth={}, t.size={}, t.get(0)={}, t.get(1)={}", cute.rank(t), cute.depth(t), cute.size(t), cute.get(t, (0,)), cute.get(t, (1,)))
```

위 코드의 출력은 다음과 같다.

```
t.rank=2, t.depth=2, t.size=24, t.get(0)=(1,2), t.get(1)=(3,4)
```

### Layout, Shape, Stride & Tensor

`Shape`와 `Stride`는 모두 `IntTuple`로 구성된다.

`Layout`은 `(Shape, Stride)`의 튜플이다. 의미론적으로, 이는 `Stride`를 통해 `Shape` 내의 모든 좌표를 인덱스로 매핑하는 기능을 구현한다.
주로 표기는 `Shape:Stride`로 표기한다.

`Layout`은 포인터나 배열과 같은 데이터와 결합하여 `Tensor`를 생성할 수 있다. Layout에 의해 생성된 인덱스는 반복자(iterator)를 참조하여 적절한 데이터를 검색하는 데 사용된다.

## Layout의 생성 및 사용

### `make_layout`: Shape/Stride로 레이아웃 만들기

Python DSL의 `cute.make_layout(shape, stride=...)`는 C++의 `make_layout(make_shape(...), make_stride(...))`에 해당한다.

`stride`를 생략할 경우, `LayoutLeft`를 기본으로 하여 생성된다. 
`LayoutLeft`는 `Shape`의 계층 구조와 상관없이 왼쪽에서 오른쪽으로 `Shape`의 배타적 접두사 곱(exclusive prefix product)으로 스트라이드를 생성한다. 이는 "일반화된 열 우선(column-major) 스트라이드 생성"으로 간주될 수 있습니다. 
`LayoutRight`는 `Shape`의 계층 구조와 상관없이 오른쪽에서 왼쪽으로 `Shape`의 배타적 접두사 곱으로 스트라이드를 생성한다. 깊이가 1인 형상(shape)의 경우 이는 "행 우선(row-major) 스트라이드 생성"으로 간주될 수 있다.
계층적 형상의 경우 결과 스트라이드가 예상과 다를 수 있다. 예를 들어, 위 s2xh4 의 스트라이드는 LayoutRight 으로 생성될 수 있다.

Python DSL에서는 `LayoutRight`로 자동으로 생성하는 옵션은 없는 듯하다. 다만 `make_ordered_layout`를 활용하면 stride를 계산하지 않고도 layout을 생성할 수 있다. 만들려고 하는 layout이 `LayoutRight`인 경우, Shape와 동일한 형태의 order를 우측부터 좌측으로 0, 1, 2, ... 순서로 지정하면 된다.

`Shape`와 `Stride`는 합동(congruent)이어야 한다. 즉, `Shape`와 `Stride`는 동일한 튜플 크기를 가진다.
`cute.is_congruent(shape, stride)`로 확인할 수 있다.

``` python
s2xd4_a = cute.make_layout((2, cute.Int32(4)), stride=(12, 1))
s2xd4_col = cute.make_layout((2, cute.Int32(4)))
s2xd4_row = cute.make_layout((2, cute.Int32(4)), stride=(4, 1))

"""
static print s2xd4_col: {} (2,?):(1,2)
static print s2xd4_row: {} (2,?):(4,1)
dynamic print s2xd4_col: (2,4):(1,2)
dynamic print s2xd4_row: (2,4):(4,1)
"""

s2xh4 = cute.make_layout((2, (2, 2)), stride=(4, (2, 1)))
s2xh4_col = cute.make_layout(s2xh4.shape)

"""
dynamic print s2xh4: (2,(2,2)):(4,(2,1))
dynamic print s2xh4_col: (2,(2,2)):(1,(2,4))
"""

s2xh4_col = cute.make_ordered_layout((2, (2, 2)), (0, (1, 2)))
s2xh4_row = cute.make_ordered_layout((2, (2, 2)), (2, (1, 0)))

"""
dynamic print s2xh4_col: (2,(2,2)):(1,(2,4))
dynamic print s2xh4_row: (2,(2,2)):(4,(2,1))
"""
```

### Layout 사용하기

`Layout`의 용도는 `Shape`에 의해 정의된 좌표 공간과 `Stride`에 의해 정의된 인덱스 공간 사이를 매핑하는 것이다.
`layout(m,n)` 구문은 논리적인 2D 좌표 (m,n)에서 1D 인덱스로의 매핑을 제공한다.

```
> print2D(s2xs4)        # (2, 4):(1, 2)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)      # (2, 4):(12, 1)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)   # (2, (2, 2)):(1, (2, 4))
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)       # (2, (2, 2)):(4, (2, 1))
  0    2    1    3
  4    6    5    7

[print2D 참고](./13-layout.py)
```

흥미롭게도, `s2xh4` 예제는 행 우선(row-major)도 열 우선(column-major)도 아니다. 
게다가 세 개의 모드를 가지고 있음에도 여전히 랭크-2로 해석되며 2D 좌표를 사용하고 있다. 
구체적으로, `s2xh4`는 두 번째 모드에 2D 멀티 모드를 가지고 있지만, 우리는 여전히 해당 모드에 대해 1D 좌표를 사용할 수 있다. 

```
> print1D(s2xs4)        # (2, 4):(1, 2)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)      # (2, 4):(12,1)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)   # (2, (2, 2)):(1, (2, 4))
  0    1    2    3    4    5    6    7
> print1D(s2xh4)       # (2, (2, 2)):(4, (2, 1))
  0    4    2    6    1    5    3    7
```

전체 레이아웃을 포함하여 레이아웃의 모든 멀티 모드(multi-mode)는 1차원 좌표를 허용한다.

### 개인적으로 경험한 Layout 좌표 생각하는 방법

Layout 좌표를 생각하는 방법은 다음과 같다.

1. 우선 1-D인지 2-D인지 확인한다.
2. 정해졌다면 해당 축에 있는 nested mode들은 그냥 flatten 시켜서 생각한다.
3. 가장 왼쪽의 mode부터 shape 갯수를 맞춰서 stride를 더해주고 하나의 tuple(?) 처럼 생각한다.
4. 갯수가 맞춰졌으면 다시 남은 것 중 가장 왼쪽의 mode를 골라 shpae 갯수를 맞춰서 tuple에 stride를 더하여 또 tuple로 만든다.
5. 더이상 남은 mode가 없어질 때까지 반복한다.
6. 해당 mode에 대해 평탄화한다.

예를 들어, (2, (2, 2)):(4, (2, 1))를 1-D 평탄화 한다고 가정해보자.

1. 1-D이다.
2. (2, 2, 2):(4, 2, 1)을 다룬다고 생각한다.
3. 가장 왼쪽 shape 2와 stride 4를 맞춰주기 위해 (0, 4)를 만든다.
4. 다음 왼쪽 shape 2와 stride 2를 맞춰주기 위해 ((0, 4), (2, 6))을 만든다.
5. 다음 왼쪽 shape 2와 stride 1을 맞춰주기 위해 (((0, 4), (2, 6)), ((1, 5), (3, 7)))을 만든다.
6. 평탄화 하여 0, 4, 2, 6, 1, 5, 3, 7을 얻는다.

비슷하게 2-D 좌표에 대해서도 각각축에 대해 적용하면 되는 것 같다.







### 참고 문헌
- [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
