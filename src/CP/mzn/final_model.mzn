include "globals.mzn";

%% INPUTS %%

int: plate;
int: n_circuits;
array[1..n_circuits, 1..2] of int: circuits;

array[1..n_circuits] of int: x=[circuits[i,1]| i in 1..n_circuits];
array[1..n_circuits] of int: y=[circuits[i,2]| i in 1..n_circuits];

int: height_min =  min(y);
int: height_max = ceil(sum(y)/2);

%% VARIABLES %%
var height_min..height_max: height;
array [1..n_circuits] of var 0..(plate-min(x)): coord_x;
array [1..n_circuits] of var 0..(height_max-min(y)): coord_y;

%% CONSTRAINTS %%
constraint diffn(coord_x, coord_y, x, y);
constraint forall(i in 1..n_circuits)(coord_x[i]+x[i]<=plate /\ coord_y[i]+y[i]<=height);
constraint cumulative(coord_y, y, x, plate);
constraint cumulative(coord_x, x, y, height);

%% SYMMETRY BREAKING %%
%symmetry breaking on axes
predicate symm_breaking_axes() =
      let {
            array[1..n_circuits] of var int: x_sim = [plate - coord_x[i] - x[i] | i in 1..n_circuits];
            array[1..n_circuits] of var int: y_sim = [height - coord_y[i] - y[i] | i in 1..n_circuits]
      } in lex_lesseq(coord_x, x_sim) /\ lex_lesseq(coord_y, y_sim);

%symmetry breaking for circuits of same height and width
predicate symm_breaking_same_circuits() =
   forall(i, j in 1..n_circuits where i<j)(
     if x[i]==x[j] /\ y[i]==y[j]
     then lex_lesseq([coord_x[i], coord_y[i]], [coord_x[j], coord_y[j]])
     endif
   );

constraint symm_breaking_same_circuits()  /\ symm_breaking_axes();


%% SEARCH %%

solve :: seq_search([
                    int_search([height], input_order, indomain_min)::restart_luby(200),
                    int_search(coord_x, input_order, indomain_min),
                    int_search(coord_y, input_order, indomain_min)
                    ])satisfy;

output["\(coord_x)\n\(coord_y)\n\(x)"]