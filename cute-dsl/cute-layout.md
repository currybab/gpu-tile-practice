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










### 참고 문헌
- [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
