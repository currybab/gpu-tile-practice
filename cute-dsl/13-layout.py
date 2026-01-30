import cutlass
import cutlass.cute as cute

@cute.jit
def print2D(layout: cute.Layout):
    # Python DSL에서는 layout(m, n)이 아니라 layout((m, n)) 형태로 호출해야 함.
    for m in cutlass.range_constexpr(cute.size(layout, mode=[0])):
        for n in cutlass.range_constexpr(cute.size(layout, mode=[1])):
            print(layout((m, n)), end="  ")
        print("")

@cute.jit
def print1D(layout: cute.Layout):
    for m in cutlass.range_constexpr(cute.size(layout)):
        print(layout(m), end="  ")
    print("")        

@cute.jit
def layout_test():
    t = ((1,2), (3,4))
    cute.printf("t.rank={}, t.depth={}, t.size={}, t.get(0)={}, t.get(1)={}", cute.rank(t), cute.depth(t), cute.size(t), cute.get(t, (0,)), cute.get(t, (1,)))
    # t.rank=2, t.depth=2, t.size=24, t.get(0)=(1,2), t.get(1)=(3,4)

    s8 = cute.make_layout(8);
    d8 = cute.make_layout(cute.Int32(8))
    cute.printf("dynamic print s8: {}", s8)
    print("static print s8: {}", s8)
    cute.printf("dynamic print d8: {}", d8)
    print("static print d8: {}", d8)

    """
    print 결과:
    static print s8: {} 8:1
    static print d8: {} ?:1
    dynamic print s8: 8:1
    dynamic print d8: 8:1
    """

    s2xs4 = cute.make_layout((2, 4));
    s2xd4 = cute.make_layout((2, cute.Int32(4)));
    cute.printf("dynamic print s2xs4: {}", s2xs4)
    cute.printf("dynamic print s2xd4: {}", s2xd4)
    print("static print s2xs4: {}", s2xs4)
    print("static print s2xd4: {}", s2xd4)

    """
    print 결과:
    static print s2xs4: {} (2,4):(1,2)
    static print s2xd4: {} (2,?):(1,2)
    dynamic print s2xs4: (2,4):(1,2)
    dynamic print s2xd4: (2,4):(1,2)
    """

    s2xd4_a = cute.make_layout((2, cute.Int32(4)), stride=(12, 1))
    s2xd4_col = cute.make_layout((2, cute.Int32(4)))
    s2xd4_row = cute.make_layout((2, cute.Int32(4)), stride=(4, 1))

    cute.printf("dynamic print s2xd4_a: {}", s2xd4_a)
    cute.printf("dynamic print s2xd4_col: {}", s2xd4_col)
    cute.printf("dynamic print s2xd4_row: {}", s2xd4_row)
    print("static print s2xd4_a: {}", s2xd4_a)
    print("static print s2xd4_col: {}", s2xd4_col)
    print("static print s2xd4_row: {}", s2xd4_row)

    """
    print 결과:
    static print s2xd4_col: {} (2,?):(1,2)
    static print s2xd4_row: {} (2,?):(4,1)
    dynamic print s2xd4_col: (2,4):(1,2)
    dynamic print s2xd4_row: (2,4):(4,1)
    """

    s2xh4 = cute.make_layout((2, (2, 2)), stride=(4, (2, 1)))
    s2xh4_col = cute.make_layout(s2xh4.shape)
    cute.printf("dynamic print s2xh4: {}", s2xh4)
    cute.printf("dynamic print s2xh4_col: {}", s2xh4_col)

    """
    print 결과:
    dynamic print s2xh4: (2,(2,2)):(4,(2,1))
    dynamic print s2xh4_col: (2,(2,2)):(1,(2,4))
    """

    s2xh4_col = cute.make_ordered_layout((2, (2, 2)), (0, (1, 2)))
    s2xh4_row = cute.make_ordered_layout((2, (2, 2)), (2, (1, 0)))
    cute.printf("dynamic print s2xh4_col: {}", s2xh4_col)
    cute.printf("dynamic print s2xh4_row: {}", s2xh4_row)


    """
    print 결과:
    dynamic print s2xh4_col: (2,(2,2)):(1,(2,4))
    dynamic print s2xh4_row: (2,(2,2)):(4,(2,1))
    """

    #####################################################

    cute.printf("congruent: {}", cute.is_congruent(s2xh4_row.shape, s2xh4_col.stride))
    
    #####################################################
    
    print("s2xs4 2D index map:") 
    print2D(s2xs4)

    print("s2xh4_col 2D index map:") # (2,(2,2)):(1,(2,4))
    print2D(s2xh4_col)

    print("s2xh4_row 2D index map:") # (2,(2,2)):(4,(2,1))
    print2D(s2xh4_row)

    #####################################################

    print("s2xs4 1D index map:") 
    print1D(s2xs4)

    print("s2xh4_col 1D index map:") # (2,(2,2)):(1,(2,4))
    print1D(s2xh4_col)

    print("s2xh4_row 1D index map:") # (2,(2,2)):(4,(2,1))
    print1D(s2xh4_row)

    #######################################################

    cute.printf("s2xh4_col 1D index map with idx2crd:") # (2,(2,2)):(1,(2,4))
    for i in range(cute.size(s2xh4_col)):
        cute.printf("1-D {} = h-D {}", i, cute.idx2crd(i, s2xh4_col.shape))

    """
    1-D 0 = h-D (0,(0,0))
    1-D 1 = h-D (1,(0,0))
    1-D 2 = h-D (0,(1,0))
    1-D 3 = h-D (1,(1,0))
    1-D 4 = h-D (0,(0,1))
    1-D 5 = h-D (1,(0,1))
    1-D 6 = h-D (0,(1,1))
    1-D 7 = h-D (1,(1,1))
    """

    cute.printf("s2xh4_col 2D index map with idx2crd:") # (2,(2,2)):(1,(2,4))
    for j in range(cute.size(s2xh4_col, mode=[1])):
        for i in range(cute.size(s2xh4_col, mode=[0])):
            cute.printf("2-D ({}, {}) = h-D {}", i, j, cute.idx2crd((i, j), s2xh4_col.shape))

    """
    2-D (0, 0) = h-D (0,(0,0))
    2-D (1, 0) = h-D (1,(0,0))
    2-D (0, 1) = h-D (0,(1,0))
    2-D (1, 1) = h-D (1,(1,0))
    2-D (0, 2) = h-D (0,(0,1))
    2-D (1, 2) = h-D (1,(0,1))
    2-D (0, 3) = h-D (0,(1,1))
    2-D (1, 3) = h-D (1,(1,1))
    """

    #######################################################

    cute.printf("s2xh4_col 1D index map with crd2idx:") # (2,(2,2)):(1,(2,4))
    for i in range(cute.size(s2xh4_col)):
        cute.printf("1-D {} = index {}", i, cute.crd2idx(i, s2xh4_col))

    """
    1-D 0 = index 0
    1-D 1 = index 1
    1-D 2 = index 2
    1-D 3 = index 3
    1-D 4 = index 4
    1-D 5 = index 5
    1-D 6 = index 6
    1-D 7 = index 7
    """    

    cute.printf("s2xh4_row 1D index map with crd2idx:") # (2,(2,2)):(4,(2,1))
    for i in range(cute.size(s2xh4_row)):
        cute.printf("1-D {} = index {}", i, cute.crd2idx(i, s2xh4_row))

    """
    1-D 0 = index 0
    1-D 1 = index 4
    1-D 2 = index 2
    1-D 3 = index 6
    1-D 4 = index 1
    1-D 5 = index 5
    1-D 6 = index 3
    1-D 7 = index 7
    """

    #######################################################
    # sublayout
    a = cute.make_ordered_layout((4, (3, 6)), (0, (1, 2)))
    a0 = cute.get(a, mode=[0])
    a1 = cute.get(a, mode=[1])
    a10 = cute.get(a, mode=[1, 0])
    a11 = cute.get(a, mode=[1, 1])
    cute.printf("a: {}", a)  # a: (4,(3,6)):(1,(4,12))
    cute.printf("a0: {}", a0)  # a0: 4:1
    cute.printf("a1: {}", a1)  # a1: (3,6):(4,12)
    cute.printf("a10: {}", a10)  # a10: 3:4
    cute.printf("a11: {}", a11)  # a11: 6:12

    a = cute.make_ordered_layout((2, 3, 5, 7), (0, 1, 2, 3))
    a13 = cute.select(a, mode=[1, 3])
    a01 = cute.select(a, mode=[0, 1, 3])
    a2 = cute.select(a, mode=[2])
    cute.printf("a: {}", a)  # a: (2,3,5,7):(1,2,6,30)  
    cute.printf("a13: {}", a13)  # a13: (3,7):(2,30)
    cute.printf("a01: {}", a01)  # a01: (2,3,7):(1,2,30)
    cute.printf("a2: {}", a2)  # a2: (5):(6)

    #######################################################
    # concatenation
    a = cute.make_layout(3, stride=1)
    b = cute.make_layout(4, stride=3)
    cute.printf("a: {}", a) # a: 3:1
    cute.printf("b: {}", b) # b: 4:3
    row = cute.append(a, b)
    cute.printf("row: {}", row) # row: (3,4):(1,3)
    col = cute.append(b, a)
    cute.printf("col: {}", col) # col: (4,3):(3,1)
    ab = cute.append(a, b)
    ba = cute.prepend(a, b)
    cute.printf("ab: {}", ab) # ab: (3,4):(1,3)
    cute.printf("ba: {}", ba) # ba: (4,3):(3,1)
    c = cute.append(ab, ab)
    cute.printf("c: {}", c) # c: (3,4,(3,4)):(1,3,(1,3))

    #######################################################
    # Grouping and flattening
    a = cute.make_ordered_layout((2, 3, 5, 7), (0, 1, 2, 3))
    cute.printf("a: {}", a) # a: (2,3,5,7):(1,2,6,30)
    b = cute.group_modes(a, 0, 2)
    cute.printf("b: {}", b) # b: ((2,3),5,7):((1,2),6,30)
    c = cute.group_modes(b, 1, 3)
    cute.printf("c: {}", c) # c: ((2,3),(5,7)):((1,2),(6,30))
    f = cute.flatten(b)
    cute.printf("f: {}", f) # f: (2,3,5,7):(1,2,6,30)
    e = cute.flatten(c)
    cute.printf("e: {}", e) # e: (2,3,5,7):(1,2,6,30)



    #######################################################
    # Slicing



    

layout_test()
