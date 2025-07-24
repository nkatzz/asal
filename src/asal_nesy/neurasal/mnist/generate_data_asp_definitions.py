diverse_models_meta_encoding = \
    """
model(1..m).

conjunction(B,M) :- model(M), literal_tuple(B),
        hold(L,M) : literal_tuple(B, L), L > 0;
    not hold(L,M) : literal_tuple(B,-L), L > 0.

body(normal(B),M) :- rule(_,normal(B)), conjunction(B,M).
body(sum(B,G),M)  :- model(M), rule(_,sum(B,G)),
    #sum { W,L :     hold(L,M), weighted_literal_tuple(B, L,W), L > 0 ;
           W,L : not hold(L,M), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

  hold(A,M) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,M).
{ hold(A,M) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B,M).

show(T,M) :- output(T,B), conjunction(B,M).
#show.
#show (T,M) : show(T,M).

#const k=1.

:- model(M), model(N), M<N, option = 1,
    #sum{ 1,T: show(T,M), not show(T,N) ;
          1,T: not show(T,M), show(T,N) } < k.

#maximize{ 1,T,M,N: show(T,M), not show(T,N), model(N), option = 2 }.
"""

sfa_interpreter = \
"""
% Automata interpreter
inState(1,T) :- seqStart(T).
inState(S2,T+1) :- inState(S1,T), transition(S1,f(S1,S2),S2), holds(f(S1,S2),T).
accepted :- inState(S,T), accepting(S), seqEnd(T).
reach_accept_at(T) :- inState(S,T), accepting(S); #false: inState(S,T1), T1 < T.

seqStart(1).
seqEnd(T+1) :- time(T), not time(T+1).
digit(0..9).

% :- reach_accept_at(T), not seqEnd(T).  % reach acceptance only at the end of the sequence

#show.
#show seq/2.
"""

time_range = lambda x: f'time(1..{x}).'

sfa_1 = \
"""
% Automaton Definition.
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,even),T), holds(equals(d1,gt_6),T).
holds(f(2,3),T) :- holds(equals(d1,odd),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,leq_3),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).

holds(f(4,4),T) :- time(T).  % No over-general self loop to avoid having acceptance prob. increase indefinitely in the experiments.
% holds(f(4,4),T) :- holds(equals(d1,leq_3),T).
% holds(f(4,1),T) :- time(T), not holds(f(4,4),T).
"""

sfa_2 = \
"""
% Automaton Definition.
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,odd),T), holds(equals(d1,gt_6),T).
holds(f(2,3),T) :- holds(equals(d1,even),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,even),T), holds(equals(d1,leq_3),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).

holds(f(4,4),T) :- time(T).  % No over-general self loop to avoid having acceptance prob. increase indefinitely in the experiments.
% holds(f(4,4),T) :- holds(equals(d1,leq_3),T).
% holds(f(4,1),T) :- time(T), not holds(f(4,4),T).
"""

# For each new pattern defined a 'name' deeds to be added here.
pattern_names = {sfa_1: 'sfa_1', sfa_2: 'sfa_2'}

predicate_defs = \
"""
% Predicate definitions
holds(equals(D,even),T) :- seq(obs(D,X),T), X \ 2 = 0.
holds(equals(D,odd),T) :- seq(obs(D,X),T), X \ 2 != 0.
holds(equals(D,gt_6),T) :- seq(obs(D,X),T), X > 6.
holds(equals(D,leq_6),T) :- seq(obs(D,X),T), X <= 6.
holds(equals(D,gt_3),T) :- seq(obs(D,X),T), X > 3.
holds(equals(D,leq_3),T) :- seq(obs(D,X),T), X <= 3.
holds(equals(D,gt_5),T) :- seq(obs(D,X),T), X > 5.
"""

def get_seq_generation_choices(seq_dim):
    choices = []
    for i in range(1, seq_dim + 1):
        choices.append(f"""1 {{seq(obs(d{i},X),T) : digit(X)}} 1 :- time(T).""")

    return '\n'.join(choices)

def get_program(seq_length, seq_dim, sfa_pattern):
    p = [sfa_interpreter, sfa_pattern, time_range(seq_length), predicate_defs, get_seq_generation_choices(seq_dim)]
    return '\n'.join(p)

if __name__ == '__main__':
    p = get_program(50, 3, sfa)
    print(p)

