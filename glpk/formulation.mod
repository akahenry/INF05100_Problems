# Number of vertices
param V integer >= 0;

# Set of colors
set COLORS := 0..(V-1);

# Set of COLORS_MINUS_1
set COLORS_MINUS_1 := 0..(V-2);

# Set of vertices
set VERTICES := 0..(V-1);

# Set of edges
param E {i in VERTICES, j in VERTICES} binary default 0;

# Represents if the color 'h' is used
var y { h in COLORS } binary;

# Represents if the color 'h' is used in vertex 'i'
var x { i in VERTICES, h in COLORS } binary;

# Minimize the number of used colors
minimize colors: sum{h in COLORS} y[h];

s.t. W1 {i in VERTICES} : sum{h in COLORS} x[i, h] = 1;

s.t. W2 {i in VERTICES, j in VERTICES, h in COLORS} : E[i, j]*(x[i, h] + x[j, h]) <= E[i, j]*y[h];

s.t. W3 {h in COLORS_MINUS_1} : sum{i in VERTICES} x[i, h] >= sum{i in VERTICES} x[i, h+1];