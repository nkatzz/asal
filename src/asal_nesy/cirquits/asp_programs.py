mnist = """
        nn_input(d1). nn_input(d2).
        1 {value(D,0..9)} 1 :- nn_input(D).
        sum(Z) :- value(D1,X), value(D2,Y), D1 != D2, Z = X + Y.
        query(sum,X) :- sum(X).  % the first argument is used to name the generated queries.
        #show value/2.
        #show query/2.
        """

mnist_even_odd = \
    """
    1 {value(d,0..9)} 1.
    even(d) :- value(d,N), N \ 2 = 0.
    odd(d) :- value(d,N), N \ 2 != 0.
    larger_than_6(d) :- value(d,N), N > 6.
    less_eq_6(d) :- value(d,N), N <= 6.
    larger_than_3(d) :- value(d,N), N > 3.
    less_eq_3(d) :- value(d,N), N <= 3.

    f(1,2) :- even(d), larger_than_6(d).               
    % f(1,2) :- larger_than_6(d).
    f(1,1) :- not f(1,2).

    f(2,3) :- odd(d), less_eq_6(d).
    % f(2,3) :- larger_than_6(d).
    f(2,2) :- not f(2,3).

    f(3,4) :- less_eq_3(d).
    f(3,3) :- not f(3,4).

    query(guard,f(1,2)) :- f(1,2).
    query(guard,f(1,1)) :- f(1,1).
    query(guard,f(2,3)) :- f(2,3).
    query(guard,f(2,2)) :- f(2,2).
    query(guard,f(3,4)) :- f(3,4).
    query(guard,f(3,3)) :- f(3,3).

    #show value/2.
    #show query/2."""

mnist_even_odd_learn = \
    """
    1 {value(d,0..9)} 1.
    even(d) :- value(d,N), N \ 2 = 0.
    odd(d) :- value(d,N), N \ 2 != 0.
    larger_than_6(d) :- value(d,N), N > 6.
    less_eq_6(d) :- value(d,N), N <= 6.
    larger_than_3(d) :- value(d,N), N > 3.
    less_eq_3(d) :- value(d,N), N <= 3.
    larger_than_2(d) :- value(d,N), N > 2.
    less_eq_2(d) :- value(d,N), N <= 2.
    larger_than_4(d) :- value(d,N), N > 4.
    less_eq_4(d) :- value(d,N), N <= 4.
    larger_than_5(d) :- value(d,N), N > 5.
    less_eq_5(d) :- value(d,N), N <= 5.
    larger_than_7(d) :- value(d,N), N > 7.
    less_eq_7(d) :- value(d,N), N <= 7.
    
    equals(even,1) :- even(d).
    equals(odd,1) :- odd(d).
    equals(gt_6,1) :- larger_than_6(d).
    equals(leq_6,1) :- less_eq_6(d).
    equals(gt_2,1) :- larger_than_2(d).
    equals(leq_2,1) :- less_eq_2(d).
    equals(gt_3,1) :- larger_than_3(d).
    equals(leq_3,1) :- less_eq_3(d).
    equals(gt_4,1) :- larger_than_4(d).
    equals(leq_4,1) :- less_eq_4(d).
    equals(gt_5,1) :- larger_than_5(d).
    equals(leq_5,1) :- less_eq_5(d).
    equals(gt_7,1) :- larger_than_7(d).
    equals(leq_7,1) :- less_eq_7(d).
    """

road_r_101_seq = """
        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,movtow) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,movtow) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

        f(1,2) :- value(a1,movaway), not f(1,3).
        f(1,2) :- value(a2,movtow), not f(1,3).
        f(1,3) :- value(a1,stop), value(l2,vehlane).
        f(1,3) :- value(a1,stop), value(l1,vehlane).
        f(1,1) :- not f(1,2), not f(1,3).

        f(2,1) :- value(a1,movtow), not f(2,3), not f(2,4).
        f(2,1) :- value(a2,stop), not f(2,3), not f(2,4).
        f(2,3) :- value(a2,movtow), not f(2,4).
        f(2,3) :- value(a1,movaway), not f(2,4).
        f(2,4) :- value(a1,stop), value(l2,incomlane).
        f(2,4) :- value(a2,movaway), value(l1,vehlane), value(l2,vehlane).
        f(2,2) :- not f(2,1), not f(2,3), not f(2,4).

        f(3,4) :- value(a1,movtow), value(l1,vehlane), value(l2,vehlane).
        f(3,4) :- value(a1,movaway), value(a2,movtow), value(l2,jun).
        f(3,3) :- not f(3,4).

        % f(4,4).

        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(1,1)) :- f(1,1).
        query(guard,f(2,1)) :- f(2,1).
        query(guard,f(2,2)) :- f(2,2).
        query(guard,f(2,3)) :- f(2,3).
        query(guard,f(2,4)) :- f(2,4).
        query(guard,f(3,4)) :- f(3,4).
        query(guard,f(3,3)) :- f(3,3).

        #show value/2.
        #show query/2.
        """

road_r = """
        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

        f(1,2) :- value(a1,movaway), value(l2, vehlane), not f(1,3).
        f(1,2) :- value(a1,stop), not f(1,3).
        f(1,3) :- value(l1,vehlane), value(l2,incomlane).
        f(1,3) :- value(l1,jun).
        f(3,1) :- value(l2,jun).
        f(2,4) :- value(l2,incomlane).
        f(2,4) :- value(a1,movaway), value(l1,vehlane).
        f(2,3) :- value(a1,stop), not f(2,4).
        f(3,3) :- not f(3,1).
        f(2,2) :- not f(2,3), not f(2,4).
        f(1,1) :- not f(1,2), not f(1,3).
        
        % f(4,4)  
        
        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(3,1)) :- f(3,1).
        query(guard,f(2,4)) :- f(2,4).
        query(guard,f(2,3)) :- f(2,3).
        query(guard,f(3,3)) :- f(3,3).
        query(guard,f(1,1)) :- f(1,1).
        query(guard,f(2,2)) :- f(2,2).
        
        #show value/2.
        #show query/2.
        """

road_r_1 = """
        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

        f(1,2) :- value(a2,other), not f(1,3).
        f(1,3) :- value(l1,other).
        f(2,3) :- value(l1,incomlane), not f(2,4).
        f(2,4) :- value(a1,movaway), value(a2,movaway).
        f(3,4) :- value(a1,stop), value(l2,incomlane).
        f(3,2) :- value(a2,other), not f(3,4).
        
        f(3,3) :- not f(3,2), not f(3,4).
        f(2,2) :- not f(2,3), not f(2,4).
        f(1,1) :- not f(1,2), not f(1,3).

        % f(4,4)  

        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(2,4)) :- f(2,4).
        query(guard,f(2,3)) :- f(2,3).
        query(guard,f(3,2)) :- f(3,2).
        query(guard,f(3,4)) :- f(3,4).
        query(guard,f(3,3)) :- f(3,3).
        query(guard,f(1,1)) :- f(1,1).
        query(guard,f(2,2)) :- f(2,2).

        #show value/2.
        #show query/2.
        """


road_r_5 = """
        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.


        f(1,2) :- value(a1,stop), not f(1,3).
        f(1,3) :- value(l1,vehlane).
        f(3,4) :- value(a1,movaway).
        f(2,4) :- value(l2,incomlane).
        f(2,3) :- value(l2,jun), not f(2,4).
        f(3,3) :- not f(3,4).
        f(2,2) :- not f(2,3), not f(2,4).
        f(1,1) :- not f(1,2), not f(1,3).

        % f(4,4)  

        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(3,4)) :- f(3,4).
        query(guard,f(2,4)) :- f(2,4).
        query(guard,f(2,3)) :- f(2,3).
        query(guard,f(3,3)) :- f(3,3).
        query(guard,f(1,1)) :- f(1,1).
        query(guard,f(2,2)) :- f(2,2).

        #show value/2.
        #show query/2.
        """

road_r_4 = """
        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

        f(1,2) :- value(l1,incomlane), not f(1,3).
        f(1,2) :- value(l1,other), not f(1,3).
        f(1,3) :- value(a2,stop).
        f(1,3) :- value(a1,movaway). 
        f(2,3) :- value(l1,incomlane), not f(2,4).
        f(2,3) :- value(l1,other), not f(2,4).
        f(2,4) :- value(a2,other), value(l1,vehlane), value(l2,vehlane).
        f(2,4) :- value(a1,stop), value(l2,incomlane).
        f(3,2) :- value(l2,other).
        f(3,2) :- value(a1,movaway).
        
        f(3,3) :- not f(3,2).
        f(2,2) :- not f(2,3), not f(2,4).
        f(1,1) :- not f(1,2), not f(1,3).

        % f(4,4)  

        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(3,2)) :- f(3,2).
        query(guard,f(2,4)) :- f(2,4).
        query(guard,f(2,3)) :- f(2,3).
        query(guard,f(3,3)) :- f(3,3).
        query(guard,f(1,1)) :- f(1,1).
        query(guard,f(2,2)) :- f(2,2).

        #show value/2.
        #show query/2.
        """
