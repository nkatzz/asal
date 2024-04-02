#script (python)
import string
import clingo
from clingo.symbol import Number

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

def get_index(symbol):
    alphabet = list(string.ascii_lowercase)
    return Number(int(alphabet.index(str(symbol))))

#end.

%------- %
% MNIST  %
%------- %
% Use either one of the blocks below. The second one is essentially a propositionalization approach.
% Comment-out the "equals" definition that is normally used (below in this file),
% since an equals/1 pred is provided here.
%*
holds(equals(even,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0.
holds(equals(odd,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0.
holds(equals(gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X > 6.
holds(equals(leq_6,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 6.
holds(equals(gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X > 3.
holds(equals(leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 3.

categorical(even; odd; gt_6; leq_6; gt_3; leq_3).
numerical(none).
value(even,1). value(odd,1). value(gt_6,1).
value(leq_6,1). value(gt_3,1). value(leq_3,1).
*%

%*
holds(equals(even_gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X > 6.
holds(equals(even_leq_6_gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X > 3, X <= 6.
holds(equals(even_leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X <= 3.

holds(equals(odd_gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X > 6.
holds(equals(odd_leq_6_gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X > 3, X <= 6.
holds(equals(odd_leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X <= 3.

categorical(even_gt_6; even_leq_6_gt_3; even_leq_3; odd_gt_6; odd_leq_6_gt_3; odd_leq_3).
numerical(none).
value(even_gt_6,1). value(even_leq_6_gt_3,1). value(even_leq_3,1).
value(odd_gt_6,1). value(odd_leq_6_gt_3,1). value(odd_leq_3,1).
*%


%------- %
% ROAD-R %
%------- %
% alphabetNum(1..10).
time_point(SeqId,T) :- seq(SeqId,_,T).
seq(SeqId,obs(type_1,X),T) :- agent_type(SeqId,agent1(X)), time_point(SeqId,T).
seq(SeqId,obs(type_2,X),T) :- agent_type(SeqId,agent2(X)), time_point(SeqId,T).
seq(SeqId,obs(action_1,X),T) :- seq(SeqId,a1(action(X)),T), X != av_ovtak, X != ovtak.
seq(SeqId,obs(action_2,X),T) :- seq(SeqId,a2(action(X)),T), X != av_ovtak, X != ovtak.
seq(SeqId,obs(location_1,X),T) :- seq(SeqId,a1(location(X)),T).
seq(SeqId,obs(location_2,X),T) :- seq(SeqId,a2(location(X)),T).

% attribute(type_1 ; type_2 ; action_1 ; action_2 ; location_1 ; location_2).
% categorical(type_1 ; type_2 ; action_1 ; action_2 ; location_1 ; location_2).
attribute(action_1 ; action_2 ; location_1 ; location_2).
categorical(action_1 ; action_2 ; location_1 ; location_2).
numerical(none).

% value(type_1,V) :- agent_type(_,agent1(V)).
% value(type_2,V) :- agent_type(_,agent2(V)).
value(action_1,V) :- seq(_,a1(action(V)),_), V != av_ovtak, V != ovtak.
value(action_2,V) :- seq(_,a2(action(V)),_), V != av_ovtak, V != ovtak.
value(location_1,V) :- seq(_,a1(location(V)),_).
value(location_2,V) :- seq(_,a2(location(V)),_).


%*
%----------%
% Weather  %
%----------%
alphabetNum(0..1).
seq(SeqId,obs(cloudy,X),T) :- seq(SeqId,cloudy(X),T).
seq(SeqId,obs(foggy,X),T) :- seq(SeqId,foggy(X),T).
seq(SeqId,obs(rainy,X),T) :- seq(SeqId,rainy(X),T).
seq(SeqId,obs(snowy,X),T) :- seq(SeqId,snowy(X),T).
seq(SeqId,obs(sunny,X),T) :- seq(SeqId,sunny(X),T).
attribute(cloudy; foggy; rainy; snowy; sunny).
numerical(none).
categorical(cloudy; foggy; rainy; snowy; sunny).
value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
*%

%*
%----------%
% Maritime %
%----------%
alphabetNum(1..8).
seq(SeqId,obs(longitude,X),T) :- seq(SeqId,longitude(X),T).
seq(SeqId,obs(latitude,X),T) :- seq(SeqId,latitude(X),T).
seq(SeqId,obs(speed,X),T) :- seq(SeqId,speed(X),T).
seq(SeqId,obs(heading,X),T) :- seq(SeqId,heading(X),T).
seq(SeqId,obs(course_over_ground,X),T) :- seq(SeqId,course_over_ground(X),T).
attribute(longitude; latitude; speed; heading; course_over_ground).
numerical(longitude; latitude; speed; heading; course_over_ground).
categorical(none).
% value(A,V) :- seq(S,obs(A,V),_), attribute(A).  % This is used in the generate part, it does not affect inference.
value(A,V) :- attribute(A), symbol(V).
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
*%


%*
%----------------------------------%
% AVG robot (all spatial features) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(px,X),T) :- seq(SeqId,px(X),T).
seq(SeqId,obs(py,X),T) :- seq(SeqId,py(X),T).
seq(SeqId,obs(pz,X),T) :- seq(SeqId,pz(X),T).
seq(SeqId,obs(ox,X),T) :- seq(SeqId,ox(X),T).
seq(SeqId,obs(oy,X),T) :- seq(SeqId,oy(X),T).
seq(SeqId,obs(oz,X),T) :- seq(SeqId,oz(X),T).
seq(SeqId,obs(ow,X),T) :- seq(SeqId,ow(X),T).
seq(SeqId,obs(vx,X),T) :- seq(SeqId,vx(X),T).
seq(SeqId,obs(vy,X),T) :- seq(SeqId,vy(X),T).
seq(SeqId,obs(vz,X),T) :- seq(SeqId,vz(X),T).

attribute(px; py; pz; ox; oy, oz; ow; vx; vy; vz).
numerical(px; py; pz; ox; oy, oz; ow; vx; vy; vz).

% attribute(px; py; ox; oy; vx; vy).
% numerical(px; py; ox; oy; vx; vy).

% attribute(px; py; ox; vx; vy).
% numerical(px; py; ox; vx; vy).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
*%


%*
%----------------------------------%
% AVG robot (latent features) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(latent_0,X),T) :- seq(SeqId,latent_0(X),T).
seq(SeqId,obs(latent_1,X),T) :- seq(SeqId,latent_1(X),T).
seq(SeqId,obs(latent_2,X),T) :- seq(SeqId,latent_2(X),T).
seq(SeqId,obs(latent_3,X),T) :- seq(SeqId,latent_3(X),T).

attribute(latent_0; latent_1; latent_2; latent_3).
numerical(latent_0; latent_1; latent_2; latent_3).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
*%

%*
%----------------------------------%
% AVG robot (discretized PCA) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(f1,X),T) :- seq(SeqId,f1(X),T).
seq(SeqId,obs(f2,X),T) :- seq(SeqId,f2(X),T).
seq(SeqId,obs(f3,X),T) :- seq(SeqId,f3(X),T).
seq(SeqId,obs(f4,X),T) :- seq(SeqId,f4(X),T).
seq(SeqId,obs(f5,X),T) :- seq(SeqId,f5(X),T).

attribute(f1; f2; f3; f4; f5).
numerical(f1; f2; f3; f4; f5).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
*%

%*
%-------------------------------------------%
% AVG robot (coords and simple events only) %
%-------------------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(px,X),T) :- seq(SeqId,px(X),T).
seq(SeqId,obs(py,X),T) :- seq(SeqId,py(X),T).
seq(SeqId,obs(idle,X),T) :- seq(SeqId,idle(X),T).
seq(SeqId,obs(linear,X),T) :- seq(SeqId,linear(X),T).
seq(SeqId,obs(rotational,X),T) :- seq(SeqId,rotational(X),T).

attribute(px; py; pz; idle; linear; rotational).
numerical(px; py; pz).
categorical(idle; linear; rotational).

value(px,V) :- symbol(V).
value(py,V) :- symbol(V).
value(pz,V) :- symbol(V).
symbol(Y) :- alphabetNum(Y).

value(idle,1). value(idle,0). value(linear,1). value(linear,0). value(rotational,1). value(rotational,0).
*%

%*
%------- %
% CAVIAR %
%------- %
alphabetNum(1..10).
seq(SeqId,obs(p1,X),T) :- seq(SeqId,p1(X),T).
seq(SeqId,obs(p2,X),T) :- seq(SeqId,p2(X),T).
seq(SeqId,obs(eucl_dist,X),T) :- seq(SeqId,eucl_dist(X),T).
seq(SeqId,obs(orient_diff,X),T) :- seq(SeqId,orient_diff(X),T).
numerical(eucl_dist; orient_diff).
categorical(p1; p2).
value(eucl_dist,V) :- symbol(V).
value(orient_diff,V) :- symbol(V).
value(p1,walking). value(p1,active). value(p1,inactive).
value(p2,walking). value(p2,active). value(p2,inactive).
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
*%

%*
%-----%
% BIO %
%-----%
alphabetNum(1..10).
seq(SeqId,obs(alive,X),T) :- seq(SeqId,alive(X),T).
seq(SeqId,obs(necrotic,X),T) :- seq(SeqId,necrotic(X),T).
seq(SeqId,obs(apoptotic,X),T) :- seq(SeqId,apoptotic(X),T).
numerical(alive; necrotic; apoptotic).
categorical(none).
value(A,V) :- attribute(A), symbol(V).
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
*%

%*
%----------------------------------
% Used for BSC ductal/lobular data.
%----------------------------------
holds(enriched(Pathway),SeqId,T) :- seq(SeqId,enriched(Pathway),T).
holds(not_enriched(Pathway),SeqId,T) :- seq(SeqId,not_enriched(Pathway),T).
pathway(P) :- seq(_, enriched(P), _).
pathway(P) :- seq(_, not_enriched(P), _).

cost(enriched(A),1) :- pathway(A).
cost(not_enriched(A),1) :- pathway(A).
*%



attribute(A) :- numerical(A).
attribute(A) :- categorical(A).
:- numerical(A), categorical(A).

% holds(decrease(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X < Y, numerical(A).
% holds(increase(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X > Y, numerical(A).

%-------------------------------------------------------------------------------------------------------------------------
% If a predicate is commented out below, in order to be excluded from the language bias,
% the corresponding choice rule and the corresponding used_attribute/1 predicate in induce.lp should also be commented-out.
%-------------------------------------------------------------------------------------------------------------------------
% holds(lt(A1,A2),SeqId,T) :- seq(SeqId,obs(A1,X),T), seq(SeqId,obs(A2,Y),T), X < Y, numerical(A1), numerical(A2), A1 != A2.
% holds(at_most(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y <= X, symbol(X).   % , value(A,X)
% holds(at_least(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y >= X, symbol(X).  % , value(A,X)
holds(equals(A,X),SeqId,T) :- seq(SeqId,obs(A,X),T), categorical(A), A != none.
% holds(neg(E,X),SeqId,T) :- seq(SeqId,obs(E,Y),T), value(E,X), Y != X.

% Costs for adding an atom to a transition rule.
% cost(lt(A1,A2),1) :- attribute(A1), attribute(A2).
% cost(at_most(A,V),1) :- value(A,V).
% cost(at_least(A,V),1) :- value(A,V).
cost(equals(A,V),1) :- value(A,V).
% cost(neg(A,V),1) :- value(A,V).

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

sequence(S) :- seq(S,_,_).
time(T) :- seq(_,_,T).
