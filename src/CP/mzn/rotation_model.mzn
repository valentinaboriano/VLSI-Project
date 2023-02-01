include "globals.mzn";

%% INPUTS %%

int: plate;
int: n_circuits;
array[1..n_circuits, 1..2] of int: circuits;

array[1..n_circuits] of int: x=[circuits[i,1]| i in 1..n_circuits];
array[1..n_circuits] of int: y=[circuits[i,2]|i in 1..n_circuits];

int: height_min =  max(plate, max(y));
int: height_max = ceil(sum(y)/2);

var height_min..height_max: height;

%% VARIABLES %%

array [1..n_circuits] of var 0..plate-min(min(x),min(y)): coord_x;
array [1..n_circuits] of var 0..height_max-min(min(x),min(y)): coord_y;
array [1..n_circuits] of var bool: rotation;
array [1..n_circuits] of var 1..max(max(x),max(y)): rot_x=[if rotation[i] then y[i] else x[i] endif| i in 1..n_circuits];
array [1..n_circuits] of var 1..max(max(x),max(y)): rot_y=[if rotation[i] then x[i] else y[i] endif| i in 1..n_circuits];

%% CONSTRAINTS %%

constraint forall(i in 1..n_circuits)(
                                      rot_x[i] + coord_x[i] <= plate /\ rot_y[i] + coord_y[i] <= height
                                      );

constraint forall(i in 1..n_circuits)(
                                      if rot_x[i] == rot_y[i] \/ rot_y[i]>plate \/ rot_x[i]>height
                                      then
                                      rotation[i] == false
                                      endif
                                      );

constraint diffn(coord_x, coord_y, rot_x, rot_y);
constraint cumulative(coord_y, rot_y, rot_x, plate);
constraint cumulative(coord_x, rot_x, rot_y, height);

%% SYMMETRY BREAKING %%

%symm breaking on x axis
predicate symm_breaking_x() =
      let {
           array[1..n_circuits] of var int: x_sim=[if coord_x[i]!= 0 then plate - coord_x[i] else 0 endif| i in 1..n_circuits];
           }
in lex_lesseq(coord_x, x_sim) /\ lex_lesseq(coord_y, reverse(coord_y));

%symm breaking on y axis
predicate symm_breaking_y() =
            let {
           array[1..n_circuits] of var int: y_sim=[if coord_y[i]!= 0 then height - coord_y[i] else 0 endif| i in 1..n_circuits];
           }
in lex_lesseq(coord_y, y_sim) /\ lex_lesseq(coord_x, reverse(coord_x));

%symm breaking for circuits of same height annd width
predicate symm_breaking_same_circuits() =
   forall(i, j in 1..n_circuits where i<j)(
     if rot_x[i]==rot_x[j] /\ rot_y[i]==rot_y[j]
     then lex_lesseq([coord_x[i], coord_y[i]], [coord_x[j], coord_y[j]])
     endif
   );

constraint symm_breaking_same_circuits()  /\ symm_breaking_x() /\ symm_breaking_y();


%% SEARCH %%
solve :: seq_search([
                    int_search([height], first_fail, indomain_min)::restart_constant(200),
                    int_search(coord_x, input_order, indomain_min),
                    int_search(coord_y, input_order, indomain_min)
                    ]) minimize height;

output["\(coord_x)\n\(coord_y)\n\(height)\n\(rot_x)\n\(rot_y)"]