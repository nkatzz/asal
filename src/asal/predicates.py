from src.asal.structs import EventTuple

"""
----------------------------------------------------------------------------------------------------------
Procedural implementations of all transition guards predicates defined at ASP side. These implementations
are used for the (optional) procedural (i.e., non-ASP, no reasoning-based) evaluation of an automaton.

NOTE:
- Any ASP predicate that will be used in such a procedural evaluation must be implemented here.
- The name of the function that implements a predicate must be identical to the predicate's name. For instance:

ASP side:
---------
holds(at_most(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y <= X, symbol(X).

Python side (here):
-------------------
def at_most(attr, val, event_tuple):
    return event_tuple.av_dict[attr] <= val
-----------------------------------------------------------------------------------------------------------
"""


def at_most(attr, val, evt: EventTuple):
    return evt.av_dict[attr] <= val


def at_least(attr, val, evt: EventTuple):
    return evt.av_dict[attr] >= val


def lt(attr_1, attr_2, evt: EventTuple):
    return evt.av_dict[attr_1] < evt.av_dict[attr_2]


def equals(attr, val, evt: EventTuple):
    return evt.av_dict[attr] == val


def neg(attr, val, evt: EventTuple):
    return evt.av_dict[attr] != val
