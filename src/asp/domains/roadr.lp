%*
%------- %
% ROAD-R %
%------- %

% allowed_actions(stop ; movtow ; movaway ; turrht ; turlft).
allowed_actions(stop ; movtow ; movaway).

% allowed_locations(incomlane ; jun ; busstop ; incomcyclane ; vehlane).
allowed_locations(incomlane ; jun ; vehlane).

time_point(SeqId,T) :- seq(SeqId,_,T).
seq(SeqId,obs(type_1,X),T) :- agent_type(SeqId,agent1(X)), time_point(SeqId,T).
seq(SeqId,obs(type_2,X),T) :- agent_type(SeqId,agent2(X)), time_point(SeqId,T).

% seq(SeqId,obs(action_1,X),T) :- seq(SeqId,a1(action(X)),T), X != av_ovtak, X != ovtak.
% seq(SeqId,obs(action_2,X),T) :- seq(SeqId,a2(action(X)),T), X != av_ovtak, X != ovtak.
% seq(SeqId,obs(location_1,X),T) :- seq(SeqId,a1(location(X)),T).
% seq(SeqId,obs(location_2,X),T) :- seq(SeqId,a2(location(X)),T).

seq(SeqId,obs(action_1,X),T) :- seq(SeqId,a1(action(X)),T), allowed_actions(X).
seq(SeqId,obs(action_2,X),T) :- seq(SeqId,a2(action(X)),T), allowed_actions(X).
% seq(SeqId,obs(action_1,other),T) :- seq(SeqId,a1(action(X)),T), not allowed_actions(X).
% seq(SeqId,obs(action_2,other),T) :- seq(SeqId,a2(action(X)),T), not allowed_actions(X).

seq(SeqId,obs(location_1,X),T) :- seq(SeqId,a1(location(X)),T), allowed_locations(X).
seq(SeqId,obs(location_2,X),T) :- seq(SeqId,a2(location(X)),T), allowed_locations(X).
% seq(SeqId,obs(location_1,other),T) :- seq(SeqId,a1(location(X)),T), not allowed_locations(X).
% seq(SeqId,obs(location_2,other),T) :- seq(SeqId,a2(location(X)),T), not allowed_locations(X).

% attribute(type_1 ; type_2 ; action_1 ; action_2 ; location_1 ; location_2).
% categorical(type_1 ; type_2 ; action_1 ; action_2 ; location_1 ; location_2).

attribute(action_1 ; action_2 ; location_1 ; location_2).
categorical(action_1 ; action_2 ; location_1 ; location_2).

% value(type_1,V) :- agent_type(_,agent1(V)).
% value(type_2,V) :- agent_type(_,agent2(V)).

value(action_1,V) :- seq(_,a1(action(V)),_), allowed_actions(V).
value(action_2,V) :- seq(_,a2(action(V)),_), allowed_actions(V).

value(action_1,other) :- seq(_,obs(action_1,other),_).
value(action_2,other) :- seq(_,obs(action_2,other),_).

value(location_1,V) :- seq(_,a1(location(V)),_), allowed_locations(V).
value(location_2,V) :- seq(_,a2(location(V)),_), allowed_locations(V).

value(location_1,other) :- seq(_,obs(location_1,other),_).
value(location_2,other) :- seq(_,obs(location_2,other),_).

holds(equals(same_lane,true),SeqId,T) :- seq(SeqId,obs(location_1,X),T), seq(SeqId,obs(location_2,X),T).
holds(equals(same_lane,false),SeqId,T) :- seq(SeqId,obs(location_1,X),T), seq(SeqId,obs(location_2,Y),T), X != Y.
holds(equals(same_direction,true),SeqId,T) :- seq(SeqId,obs(action_1,movtow),T), seq(SeqId,obs(action_2,movtow),T).
holds(equals(same_direction,true),SeqId,T) :- seq(SeqId,obs(action_1,movaway),T), seq(SeqId,obs(action_2,movaway),T).
holds(equals(same_direction,false),SeqId,T) :- seq(SeqId,obs(action_1,movtow),T), seq(SeqId,obs(action_2,movaway),T).
holds(equals(same_direction,false),SeqId,T) :- seq(SeqId,obs(action_1,movaway),T), seq(SeqId,obs(action_2,movtow),T).

holds(equals(behind_v1_v2,true),SeqId,T) :-
     holds(equals(same_lane,true),SeqId,T),
     seq(SeqId, a1(ycoord(y2, Y1)), T),
     seq(SeqId, a2(ycoord(y2, Y2)), T),
     Y1 < Y2.

holds(equals(in_front_v1_v2,true),SeqId,T) :-
     holds(equals(same_lane,true),SeqId,T),
     seq(SeqId, a1(ycoord(y2, Y1)), T),
     seq(SeqId, a2(ycoord(y2, Y2)), T),
     Y1 > Y2.

attribute(same_lane ; same_direction ; behind_v1_v2 ; in_front_v1_v2).
categorical(same_lane ; same_direction ; behind_v1_v2 ; in_front_v1_v2).
value(same_lane,true). value(same_direction,true). value(behind_v1_v2,true). value(in_front_v1_v2,true).
value(same_lane,false). value(same_direction,false).
numerical(none).
*%






%------- %
% ROAD-R %
%------- %
allowed_actions(stop ; movtow ; movaway).
allowed_locations(incomlane ; jun ; vehlane).


time_point(SeqId,T) :- seq(SeqId,_,T).
seq(SeqId,obs(type_1,X),T) :- agent_type(SeqId,agent1(X)), time_point(SeqId,T).
seq(SeqId,obs(type_2,X),T) :- agent_type(SeqId,agent2(X)), time_point(SeqId,T).

seq(SeqId,obs(action_1,X),T) :- seq(SeqId,a1(action(X)),T), allowed_actions(X).
seq(SeqId,obs(action_2,X),T) :- seq(SeqId,a2(action(X)),T), allowed_actions(X).

seq(SeqId,obs(location_1,X),T) :- seq(SeqId,a1(location(X)),T), allowed_locations(X).
seq(SeqId,obs(location_2,X),T) :- seq(SeqId,a2(location(X)),T), allowed_locations(X).


attribute(action_1 ; action_2).
categorical(action_1 ; action_2).

value(action_1,V) :- seq(_,a1(action(V)),_), allowed_actions(V).
value(action_2,V) :- seq(_,a2(action(V)),_), allowed_actions(V).


% v1_x1 <= v2_x1
holds(equals(lt_v1x1_v2x1,true),S,T) :-
    seq(S, a1(xcoord(x1, V1x1)), T),
    seq(S, a2(xcoord(x1, V2x1)), T),
    V1x1 <= V2x1.

% v2_x1 <= v1_x1
holds(equals(lt_v2x1_v1x1,true),S,T) :-
    seq(S, a1(xcoord(x1, V1x1)), T),
    seq(S, a2(xcoord(x1, V2x1)), T),
    V2x1 <= V1x1.

% v1_x2 <= v2_x2
holds(equals(lt_v1x2_v2x2,true),S,T) :-
    seq(S, a1(xcoord(x2, V1x2)), T),
    seq(S, a2(xcoord(x2, V2x2)), T),
    V1x2 <= V2x2.

% v2_x2 <= v1_x2
holds(equals(lt_v2x2_v1x2,true),S,T) :-
    seq(S, a1(xcoord(x2, V1x2)), T),
    seq(S, a2(xcoord(x2, V2x2)), T),
    V2x2 <= V1x2.
%------------------------------------------
% v1_y1 <= v2_y1
holds(equals(lt_v1y1_v2y1,true),S,T) :-
    seq(S, a1(ycoord(y1, V1y1)), T),
    seq(S, a2(ycoord(y1, V2y1)), T),
    V1y1 <= V2y1.

% v2_y1 <= v1_y1
holds(equals(lt_v2y1_v1y1,true),S,T) :-
    seq(S, a1(ycoord(y1, V1y1)), T),
    seq(S, a2(ycoord(y1, V2y1)), T),
    V2y1 <= V1y1.

% v1_y2 <= v2_y2
holds(equals(lt_v1y2_v2y2,true),S,T) :-
    seq(S, a1(ycoord(y2, V1y2)), T),
    seq(S, a2(ycoord(y2, V2y2)), T),
    V1y2 <= V2y2.

% v2_y2 <= v1_y2
holds(equals(lt_v2y2_v1y2,true),S,T) :-
    seq(S, a1(ycoord(y2, V1y2)), T),
    seq(S, a2(ycoord(y2, V2y2)), T),
    V2y2 <= V1y2.


holds(equals(same_lane,true),SeqId,T) :- seq(SeqId,obs(location_1,X),T), seq(SeqId,obs(location_2,X),T).

attribute(same_lane ; lt_v1x1_v2x1 ; lt_v2x1_v1x1 ; lt_v1x2_v2x2; lt_v2x2_v1x2; lt_v1y1_v2y1; lt_v2y1_v1y1; lt_v1y2_v2y2; lt_v2y2_v1y2).
categorical(same_lane ; lt_v1x1_v2x1 ; lt_v2x1_v1x1 ; lt_v1x2_v2x2; lt_v2x2_v1x2; lt_v1y1_v2y1; lt_v2y1_v1y1; lt_v1y2_v2y2; lt_v2y2_v1y2).
value(same_lane,true). value(lt_v1x1_v2x1,true). value(lt_v2x1_v1x1,true). value(lt_v1x2_v2x2,true).
value(lt_v2x2_v1x2,true). value(lt_v1y1_v2y1,true). value(lt_v2y1_v1y1,true). value(lt_v1y2_v2y2,true).
value(lt_v2y2_v1y2,true).
numerical(none).
