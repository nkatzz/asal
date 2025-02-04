from src.asal.tester import test_model
from src.asal.structs import Automaton
from src.asal.auxils import get_pickled_fsm
from learntools import rewrite_automaton

"""Contains useful cases of running examples encapsulated into methods."""


def eval_automata():
    # tps: 1766, fps: 72, fns: 47, f1: 0.967
    fsm_good = """accepting(5).
             transition(1,f(1,1),1). transition(1,f(1,3),3). transition(3,f(3,3),3). transition(3,f(3,5),5). transition(5,f(5,5),5).
             holds(f(1,3),S,T) :- holds(at_most(latitude,b),S,T).
             holds(f(3,5),S,T) :- holds(at_least(latitude,c),S,T).
             holds(f(1,3),S,T) :- holds(at_most(longitude,e),S,T), holds(at_least(longitude,e),S,T), holds(at_most(heading,e),S,T).
             holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,5),S,T).
             holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,3),S,T).
             holds(f(5,5),S,T) :- sequence(S), time(T)."""

    # tps: 1627, fps: 42, fns: 186, f1: 0.934
    fsm_1 = """accepting(5).
             transition(1,f(1,1),1). transition(1,f(1,3),3). transition(3,f(3,3),3). transition(3,f(3,5),5). transition(1,f(1,4),4). transition(4,f(4,5),5). transition(5,f(5,5),5).
             holds(f(1,3),S,T) :- holds(at_most(latitude,b),S,T), not holds(f(1,4),S,T).
             holds(f(3,5),S,T) :- holds(at_least(latitude,c),S,T).
             holds(f(1,4),S,T) :- holds(at_most(longitude,c),S,T).
             holds(f(4,5),S,T) :- holds(at_least(longitude,d),S,T).
             holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,5),S,T).
             holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,3),S,T).
             holds(f(4,4),S,T) :- sequence(S), time(T), not holds(f(4,5),S,T).
             holds(f(5,5),S,T) :- sequence(S), time(T)."""

    # tps: 1709, fps: 20, fns: 104, f1: 0.964
    fsm_2 = """accepting(5).
             transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(2,f(2,2),2). transition(2,f(2,5),5). 
             transition(3,f(3,2),2). transition(3,f(3,3),3). transition(5,f(5,5),5).
             holds(f(1,2),S,T) :- holds(at_most(latitude,b),S,T), not holds(f(1,3),S,T).
             holds(f(2,5),S,T) :- holds(at_least(latitude,c),S,T).
             holds(f(1,3),S,T) :- holds(at_most(longitude,d),S,T).
             holds(f(3,2),S,T) :- holds(at_least(longitude,e),S,T).
             holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,5),S,T).
             holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,2),S,T).
             holds(f(5,5),S,T) :- sequence(S), time(T).
             holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T)."""

    # tps: 1542, fps: 55, fns: 271, f1: 0.904
    fsm_3 = """transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). transition(2,f(2,5),5). transition(5,f(5,5),5). accepting(5).
             holds(f(1,2),S,T) :- holds(at_most(latitude,b),S,T).
             holds(f(2,5),S,T) :- holds(at_least(latitude,c),S,T).
             holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T).
             holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,5),S,T).
             holds(f(5,5),S,T) :- sequence(S), time(T)."""

    # tps: 1775, fps: 15, fns: 38, f1: 0.985
    fsm_4 = """accepting(5).
               transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(2,f(2,2),2). 
               transition(2,f(2,5),5). 
               transition(3,f(3,3),3). transition(3,f(3,5),5). transition(5,f(5,5),5).
               holds(f(1,2),S,T) :- holds(at_most(latitude,b),S,T), not holds(f(1,3),S,T).
               holds(f(2,5),S,T) :- holds(at_least(latitude,c),S,T).
               holds(f(1,3),S,T) :- holds(at_most(longitude,e),S,T).
               holds(f(3,5),S,T) :- holds(at_least(longitude,f),S,T).
               holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,5),S,T).
               holds(f(5,5),S,T) :- sequence(S), time(T).
               holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,5),S,T).
               holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T)."""

    # This should be checked. Uncomment the two transitions to see what happens.
    fsm_5 = """accepting(6).
               transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(2,f(2,2),2). 
               transition(2,f(2,6),6). transition(3,f(3,2),2). transition(3,f(3,3),3). transition(6,f(6,6),6).
               holds(f(1,2),S,T) :- holds(at_most(latitude,b),S,T), not holds(f(1,3),S,T).
               holds(f(2,6),S,T) :- holds(at_least(latitude,d),S,T).
               % holds(f(1,3),S,T) :- holds(at_most(latitude,d),S,T).
               % holds(f(3,2),S,T) :- holds(at_least(latitude,e),S,T).
               holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T).
               holds(f(6,6),S,T) :- sequence(S), time(T).
               holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,2),S,T).
               holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,6),S,T)."""

    # TPs, FPs, FNs, F1: 93, 0, 19, 0.9073170731707317
    fsm_6 = """accepting(3).
               transition(1,f(1,2),2). transition(2,f(2,3),3). transition(1,f(1,3),3). transition(1,f(1,1),1). 
               transition(2,f(2,2),2). transition(3,f(3,3),3).
               holds(f(1,2),S,T) :- holds(equals(p1,walking),S,T), not holds(f(1,3),S,T).
               holds(f(2,3),S,T) :- holds(equals(p1,active),S,T), holds(at_most(orient_diff,d),S,T).
               holds(f(1,3),S,T) :- holds(at_least(orient_diff,f),S,T), holds(equals(p1,active),S,T).
               holds(f(2,3),S,T) :- holds(at_least(orient_diff,g),S,T).
               holds(f(1,3),S,T) :- holds(at_most(orient_diff,a),S,T).
               holds(f(1,2),S,T) :- holds(equals(p1,inactive),S,T), not holds(f(1,3),S,T).
               holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T).
               holds(f(3,3),S,T) :- sequence(S), time(T).
               holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,3),S,T)."""

    fsm_7 = """accepting(3).
               transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). transition(2,f(2,3),3). transition(3,f(3,3),3).
               holds(f(1,2),S,T) :- holds(at_least(orient_diff,d),S,T).
               holds(f(2,3),S,T) :- holds(at_least(orient_diff,g),S,T).
               holds(f(2,3),S,T) :- holds(at_most(orient_diff,d),S,T), holds(equals(p2,active),S,T).
               holds(f(1,2),S,T) :- holds(equals(p2,inactive),S,T).
               holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T).
               holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,3),S,T).
               holds(f(3,3),S,T) :- sequence(S), time(T)."""

    fsm_8 = """accepting(10).
transition(1,f(1,5),5). transition(2,f(2,6),6). transition(3,f(3,4),4). transition(3,f(3,10),10). transition(4,f(4,3),3). transition(5,f(5,3),3). transition(5,f(5,10),10). transition(6,f(6,9),9). transition(6,f(6,10),10). transition(7,f(7,10),10). transition(8,f(8,6),6). transition(8,f(8,10),10). transition(9,f(9,6),6). transition(1,f(1,8),8). transition(2,f(2,5),5). transition(2,f(2,9),9). transition(2,f(2,10),10). transition(3,f(3,9),9). transition(5,f(5,2),2). transition(6,f(6,4),4). transition(8,f(8,9),9). transition(9,f(9,4),4). transition(9,f(9,10),10). transition(1,f(1,3),3). transition(1,f(1,7),7). transition(3,f(3,6),6). transition(6,f(6,3),3). transition(9,f(9,8),8). transition(4,f(4,7),7). transition(7,f(7,5),5). transition(3,f(3,8),8). transition(5,f(5,4),4). transition(9,f(9,7),7). transition(6,f(6,2),2). transition(7,f(7,1),1). transition(1,f(1,4),4). transition(1,f(1,10),10). transition(2,f(2,1),1). transition(2,f(2,7),7). transition(3,f(3,7),7). transition(4,f(4,6),6). transition(7,f(7,3),3). transition(8,f(8,5),5). transition(8,f(8,7),7). transition(9,f(9,3),3). transition(2,f(2,3),3). transition(6,f(6,1),1). transition(8,f(8,1),1). transition(1,f(1,6),6). transition(3,f(3,1),1). transition(2,f(2,4),4). transition(4,f(4,2),2). transition(5,f(5,7),7). transition(7,f(7,2),2). transition(8,f(8,2),2). transition(8,f(8,4),4). transition(9,f(9,2),2). transition(2,f(2,8),8). transition(5,f(5,6),6). transition(7,f(7,4),4). transition(7,f(7,9),9). transition(9,f(9,5),5). transition(4,f(4,8),8). transition(6,f(6,8),8). transition(7,f(7,6),6). transition(8,f(8,3),3). transition(3,f(3,2),2). transition(4,f(4,10),10). transition(4,f(4,9),9). transition(6,f(6,5),5). transition(5,f(5,8),8). transition(5,f(5,9),9). transition(6,f(6,7),7). transition(7,f(7,8),8). transition(3,f(3,5),5). transition(5,f(5,1),1). transition(4,f(4,5),5). transition(1,f(1,2),2). transition(4,f(4,1),1). transition(9,f(9,1),1). transition(1,f(1,9),9). transition(2,f(2,2),2). transition(3,f(3,3),3). transition(4,f(4,4),4). transition(5,f(5,5),5). transition(6,f(6,6),6). transition(7,f(7,7),7). transition(8,f(8,8),8). transition(9,f(9,9),9). transition(10,f(10,10),10).
holds(f(1,5),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p2,walking),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(2,6),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p1,walking),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(3,4),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(3,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), holds(equals(p2,walking),S,T).
holds(f(4,3),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(equals(p1,active),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,3),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(5,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(equals(p1,inactive),S,T).
holds(f(6,9),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(6,10),S,T).
holds(f(6,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p2,active),S,T).
holds(f(7,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(equals(p1,active),S,T), holds(equals(p2,inactive),S,T).
holds(f(8,6),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(8,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,active),S,T), holds(equals(p2,walking),S,T).
holds(f(9,6),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(1,8),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,active),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(2,5),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,f),S,T), holds(at_most(orient_diff,f),S,T), holds(equals(p1,walking),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(2,9),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(2,10),S,T).
holds(f(2,10),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), holds(equals(p2,walking),S,T).
holds(f(3,9),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(3,10),S,T).
holds(f(5,2),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(equals(p1,walking),S,T), holds(at_least(orient_diff,f),S,T), not holds(f(5,3),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,4),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(8,9),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(8,10),S,T).
holds(f(9,4),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p2,walking),S,T), holds(equals(p1,active),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(9,10),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T).
holds(f(1,3),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_least(orient_diff,d),S,T), holds(equals(p2,walking),S,T), not holds(f(1,4),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,7),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_most(eucl_dist,g),S,T), holds(equals(p1,walking),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(3,6),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_least(orient_diff,f),S,T), holds(at_most(orient_diff,f),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(6,3),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(9,8),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_most(eucl_dist,g),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(9,10),S,T).
holds(f(4,7),S,T) :- holds(at_least(eucl_dist,h),S,T), holds(at_most(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(7,5),S,T) :- holds(at_least(eucl_dist,h),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(3,8),S,T) :- holds(at_least(eucl_dist,i),S,T), holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(lt(eucl_dist,orient_diff),S,T), holds(equals(p1,walking),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(5,4),S,T) :- holds(at_least(eucl_dist,i),S,T), holds(at_most(orient_diff,c),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(9,7),S,T) :- holds(at_least(eucl_dist,i),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p1,walking),S,T), holds(at_least(orient_diff,d),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(6,2),S,T) :- holds(at_least(eucl_dist,j),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(6,3),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(7,1),S,T) :- holds(at_least(eucl_dist,j),S,T), holds(at_most(orient_diff,c),S,T), not holds(f(7,2),S,T), not holds(f(7,3),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(1,4),S,T) :- holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), holds(equals(p1,inactive),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,10),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,walking),S,T), holds(lt(eucl_dist,orient_diff),S,T).
holds(f(2,1),S,T) :- holds(at_least(orient_diff,d),S,T), holds(equals(p1,inactive),S,T), not holds(f(2,3),S,T), not holds(f(2,4),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(2,7),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(lt(eucl_dist,orient_diff),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(3,7),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,h),S,T), holds(equals(p2,walking),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(4,6),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,g),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(7,3),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(8,5),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(equals(p1,walking),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(8,7),S,T) :- holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(9,3),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,f),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(2,3),S,T) :- holds(at_least(orient_diff,g),S,T), holds(at_most(eucl_dist,g),S,T), not holds(f(2,4),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(6,1),S,T) :- holds(at_least(orient_diff,g),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(6,2),S,T), not holds(f(6,3),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(8,1),S,T) :- holds(at_least(orient_diff,g),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(8,2),S,T), not holds(f(8,3),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(1,6),S,T) :- holds(at_least(eucl_dist,d),S,T), holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(3,1),S,T) :- holds(at_least(eucl_dist,d),S,T), holds(equals(p1,walking),S,T), holds(at_most(eucl_dist,d),S,T), not holds(f(3,2),S,T), not holds(f(3,4),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(1,4),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(equals(p2,walking),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,5),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(equals(p1,active),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(2,4),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(2,7),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(3,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T).
holds(f(4,2),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), not holds(f(4,3),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,7),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,h),S,T), holds(lt(eucl_dist,orient_diff),S,T), holds(equals(p2,walking),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,9),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p2,inactive),S,T), not holds(f(6,10),S,T).
holds(f(7,2),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p1,active),S,T), not holds(f(7,3),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(7,3),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(8,2),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,inactive),S,T), not holds(f(8,3),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(8,4),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(8,5),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(8,10),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(equals(p1,walking),S,T).
holds(f(9,2),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(9,3),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(9,3),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(9,6),S,T) :- holds(at_least(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(1,10),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), holds(equals(p1,active),S,T).
holds(f(2,8),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,d),S,T), holds(lt(eucl_dist,orient_diff),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(3,7),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(4,6),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,6),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,1),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(orient_diff,e),S,T), holds(equals(p2,walking),S,T), not holds(f(6,2),S,T), not holds(f(6,3),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(7,4),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,e),S,T), holds(at_most(eucl_dist,f),S,T), holds(equals(p2,walking),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(7,9),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,e),S,T), holds(at_most(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(7,10),S,T).
holds(f(8,9),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_most(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(8,10),S,T).
holds(f(9,5),S,T) :- holds(at_least(eucl_dist,f),S,T), holds(at_least(orient_diff,f),S,T), holds(at_most(orient_diff,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(3,6),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_most(eucl_dist,g),S,T), holds(equals(p2,inactive),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(4,8),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(6,8),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(at_least(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(7,6),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(8,3),S,T) :- holds(at_least(eucl_dist,g),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,active),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(3,2),S,T) :- holds(at_least(eucl_dist,h),S,T), holds(at_most(eucl_dist,h),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(3,4),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(4,10),S,T) :- holds(at_least(eucl_dist,i),S,T), holds(at_most(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(at_least(orient_diff,e),S,T).
holds(f(5,4),S,T) :- holds(at_least(eucl_dist,i),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), holds(at_least(orient_diff,f),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(4,9),S,T) :- holds(at_least(eucl_dist,j),S,T), holds(at_most(orient_diff,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(4,10),S,T).
holds(f(6,5),S,T) :- holds(at_least(eucl_dist,j),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(2,10),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T).
holds(f(3,9),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(3,10),S,T).
holds(f(5,8),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,d),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(5,9),S,T) :- holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(5,10),S,T).
holds(f(6,4),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,active),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(6,7),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(6,10),S,T) :- holds(at_least(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), holds(equals(p2,walking),S,T).
holds(f(7,8),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(9,4),S,T) :- holds(at_least(orient_diff,d),S,T), holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(3,5),S,T) :- holds(at_least(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(8,6),S,T) :- holds(at_least(orient_diff,h),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(5,1),S,T) :- holds(at_least(orient_diff,j),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(5,2),S,T), not holds(f(5,3),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(2,4),S,T) :- holds(at_most(eucl_dist,c),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(4,5),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(4,9),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(4,10),S,T).
holds(f(5,7),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,5),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(equals(p1,walking),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(9,5),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(equals(p1,walking),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(1,2),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p2,walking),S,T), not holds(f(1,3),S,T), not holds(f(1,4),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(2,8),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(lt(eucl_dist,orient_diff),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(3,5),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(4,8),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,inactive),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,6),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(5,8),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), holds(equals(p2,walking),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,7),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(6,8),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(7,9),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(7,10),S,T).
holds(f(9,2),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(9,3),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(4,1),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(4,2),S,T), not holds(f(4,3),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,9),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(5,10),S,T).
holds(f(7,8),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(equals(p2,walking),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(8,4),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(9,1),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,inactive),S,T), not holds(f(9,2),S,T), not holds(f(9,3),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(1,9),S,T) :- holds(at_most(eucl_dist,f),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,active),S,T), not holds(f(1,10),S,T).
holds(f(7,2),S,T) :- holds(at_most(eucl_dist,f),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(7,3),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(7,4),S,T) :- holds(at_most(eucl_dist,f),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(7,6),S,T) :- holds(at_most(eucl_dist,g),S,T), holds(at_most(orient_diff,c),S,T), holds(equals(p1,walking),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(8,3),S,T) :- holds(at_most(eucl_dist,h),S,T), holds(at_most(orient_diff,g),S,T), holds(equals(p2,walking),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(4,10),S,T) :- holds(at_most(eucl_dist,i),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T).
holds(f(8,2),S,T) :- holds(at_most(eucl_dist,i),S,T), not holds(f(8,3),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(5,1),S,T) :- holds(at_most(orient_diff,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), not holds(f(5,2),S,T), not holds(f(5,3),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(3,2),S,T) :- holds(at_most(orient_diff,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(3,4),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(3,1),S,T) :- holds(at_most(orient_diff,f),S,T), holds(equals(p1,inactive),S,T), holds(equals(p2,active),S,T), not holds(f(3,2),S,T), not holds(f(3,4),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(2,6),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(at_most(orient_diff,d),S,T), holds(equals(p1,walking),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(4,5),S,T) :- holds(at_most(eucl_dist,c),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(4,7),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(at_most(orient_diff,e),S,T), holds(equals(p1,active),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(5,3),S,T) :- holds(at_most(eucl_dist,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(7,10),S,T) :- holds(at_most(eucl_dist,c),S,T).   
holds(f(1,2),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(1,3),S,T), not holds(f(1,4),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,3),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(equals(p2,walking),S,T), not holds(f(1,4),S,T), not holds(f(1,5),S,T), not holds(f(1,6),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,8),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,walking),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(2,1),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T), not holds(f(2,3),S,T), not holds(f(2,4),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(8,1),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(8,2),S,T), not holds(f(8,3),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(9,10),S,T) :- holds(at_most(eucl_dist,d),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p2,inactive),S,T).
holds(f(3,4),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(3,8),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(5,2),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(equals(p2,walking),S,T), not holds(f(5,3),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(6,2),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(6,3),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(6,3),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(7,1),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(lt(orient_diff,eucl_dist),S,T), not holds(f(7,2),S,T), not holds(f(7,3),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(9,1),S,T) :- holds(at_most(eucl_dist,e),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(9,2),S,T), not holds(f(9,3),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(4,3),S,T) :- holds(at_most(eucl_dist,f),S,T), holds(equals(p2,walking),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(1,7),S,T) :- holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(1,9),S,T) :- holds(at_most(orient_diff,c),S,T), holds(equals(p1,active),S,T), not holds(f(1,10),S,T).
holds(f(2,3),S,T) :- holds(at_most(orient_diff,c),S,T), holds(equals(p2,inactive),S,T), not holds(f(2,4),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(2,9),S,T) :- holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,active),S,T), not holds(f(2,10),S,T).
holds(f(7,5),S,T) :- holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), holds(equals(p2,walking),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(9,8),S,T) :- holds(at_most(orient_diff,c),S,T), holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), not holds(f(9,10),S,T).
holds(f(4,2),S,T) :- holds(lt(orient_diff,eucl_dist),S,T), not holds(f(4,3),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(2,5),S,T) :- holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,active),S,T), holds(equals(p2,inactive),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(5,10),S,T) :- holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T).
holds(f(8,7),S,T) :- holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), holds(equals(p2,walking),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(9,7),S,T) :- holds(lt(orient_diff,eucl_dist),S,T), holds(equals(p1,inactive),S,T), holds(equals(p2,walking),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T).
holds(f(1,6),S,T) :- holds(equals(p1,walking),S,T), holds(equals(p2,inactive),S,T), not holds(f(1,7),S,T), not holds(f(1,8),S,T), not holds(f(1,9),S,T), not holds(f(1,10),S,T).
holds(f(4,1),S,T) :- holds(equals(p1,active),S,T), holds(equals(p2,inactive),S,T), not holds(f(4,2),S,T), not holds(f(4,3),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,1),S,T), not holds(f(3,2),S,T), not holds(f(3,4),S,T), not holds(f(3,5),S,T), not holds(f(3,6),S,T), not holds(f(3,7),S,T), not holds(f(3,8),S,T), not holds(f(3,9),S,T), not holds(f(3,10),S,T).
holds(f(6,6),S,T) :- sequence(S), time(T), not holds(f(6,1),S,T), not holds(f(6,2),S,T), not holds(f(6,3),S,T), not holds(f(6,4),S,T), not holds(f(6,5),S,T), not holds(f(6,7),S,T), not holds(f(6,8),S,T), not holds(f(6,9),S,T), not holds(f(6,10),S,T).
holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,1),S,T), not holds(f(2,3),S,T), not holds(f(2,4),S,T), not holds(f(2,5),S,T), not holds(f(2,6),S,T), not holds(f(2,7),S,T), not holds(f(2,8),S,T), not holds(f(2,9),S,T), not holds(f(2,10),S,T).
holds(f(5,5),S,T) :- sequence(S), time(T), not holds(f(5,1),S,T), not holds(f(5,2),S,T), not holds(f(5,3),S,T), not holds(f(5,4),S,T), not holds(f(5,6),S,T), not holds(f(5,7),S,T), not holds(f(5,8),S,T), not holds(f(5,9),S,T), not holds(f(5,10),S,T).
holds(f(8,8),S,T) :- sequence(S), time(T), not holds(f(8,1),S,T), not holds(f(8,2),S,T), not holds(f(8,3),S,T), not holds(f(8,4),S,T), not holds(f(8,5),S,T), not holds(f(8,6),S,T), not holds(f(8,7),S,T), not holds(f(8,9),S,T), not holds(f(8,10),S,T).
holds(f(7,7),S,T) :- sequence(S), time(T), not holds(f(7,1),S,T), not holds(f(7,2),S,T), not holds(f(7,3),S,T), not holds(f(7,4),S,T), not holds(f(7,5),S,T), not holds(f(7,6),S,T), not holds(f(7,8),S,T), not holds(f(7,9),S,T), not holds(f(7,10),S,T).
holds(f(10,10),S,T) :- sequence(S), time(T).
holds(f(4,4),S,T) :- sequence(S), time(T), not holds(f(4,1),S,T), not holds(f(4,2),S,T), not holds(f(4,3),S,T), not holds(f(4,5),S,T), not holds(f(4,6),S,T), not holds(f(4,7),S,T), not holds(f(4,8),S,T), not holds(f(4,9),S,T), not holds(f(4,10),S,T).
holds(f(9,9),S,T) :- sequence(S), time(T), not holds(f(9,1),S,T), not holds(f(9,2),S,T), not holds(f(9,3),S,T), not holds(f(9,4),S,T), not holds(f(9,5),S,T), not holds(f(9,6),S,T), not holds(f(9,7),S,T), not holds(f(9,8),S,T), not holds(f(9,10),S,T)."""

    fsm_9 = """accepting(5).
transition(1,f(1,1),1). transition(1,f(1,5),5). transition(5,f(5,5),5).
holds(f(1,5),S,T) :- holds(equals(event,request),S,T).
holds(f(1,5),S,T) :- holds(equals(event,process_end),S,T).
holds(f(5,5),S,T) :- sequence(S), time(T).
holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,5),S,T)."""

    test_path = '/home/nkatz/n_seqs.txt'
    # test_path = '/media/nkatz/storage/seqs/caviar/caviar_data/fold1/test_fold_1_discretized.txt'
    # test_path = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TEST_SAX_8_ASP.csv'

    test_model(fsm_9, test_path, '1', 100, path_scoring=True)


def evaluate_pickled(path_to_pickled: str, data_path: str):
    fsm: Automaton = get_pickled_fsm(path_to_pickled)
    print(f'Automaton is:\n{fsm.to_string}')
    test_model(fsm.to_string, data_path, '1', 100, 5)


def show_pickled(path_to_pickled: str):
    fsm: Automaton = get_pickled_fsm(path_to_pickled)
    print(f'automaton:\n{fsm.to_string}')
    print(f'automaton.rules:\n{fsm.__rules_dict}')
    print(f'automaton.rules_mutex_conditions:\n{fsm.rules_mutex_conditions}')
    print(f'automaton.self_loop_guards:\n{fsm.self_loop_guards}')
    # Unfortunately we can't get the transitions as clingo objects in the pickled object.
    # We get them as strings to use as a surrogate for debugging.
    print(f'automaton.transitions_str:\n{fsm.transitions_str}')
    print(f'automaton.accepting_states:\n{fsm.accepting_states}')
    return fsm


def rewrite_fsm(path_to_pickled: str):
    """Debug rewrite_automaton method"""
    fsm: Automaton = show_pickled(path_to_pickled)
    print(fsm.transitions)
    _new_fsm, _map = rewrite_automaton(fsm)
    return _new_fsm, _map


if __name__ == "__main__":
    eval_automata()

    # evaluate_pickled('debug/pickled_fsms/rules-size-3',
    #                 '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TRAIN_SAX_8_ASP.csv')

    # show_pickled('debug/pickled_fsms/rules-size-5')

    # _new_fsm, _map = rewrite_fsm('debug/pickled_fsms/rules-size-5')
    # print(_new_fsm)

    """
    state_seq = ['1', '3', '1', '3', '1', '3', '1', '3', '1', '3', '1', '3', '1', '3', '1', '3', '1', '3',
                 '1', '3', '1', '3', '1', '5']

    graph = get_graph_from_state_seq(state_seq)
    print(graph)
    cycles = get_cycles_dfs(graph)
    print(cycles)
    """
