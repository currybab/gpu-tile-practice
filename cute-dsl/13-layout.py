import cutlass
import cutlass.cute as cute

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
    cute.printf("congruent: {}", cute.is_congruent(s2xh4_row.shape, s2xh4_col.stride))

layout_test()
