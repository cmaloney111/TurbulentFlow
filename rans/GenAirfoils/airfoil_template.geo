POINTS

// NOTE: For certain dat files, to avoid Vanishing Jacobian errors,
// We need to combine the two points on the trailing edge that are close together
// into one point at 1, 0. In case some airfoils are not centered, this fix may not work
// for all airfoils.

Spline(1000) = {1000:LAST_POINT_INDEX,1000};

edge_lc = 0.2;
Point(1900) = { 5, 5, 0, edge_lc};
Point(1901) = { 5, -5, 0, edge_lc};
Point(1902) = { -5, -5, 0, edge_lc};
Point(1903) = { -5, 5, 0, edge_lc};

Line(1) = {1900,1901};
Line(2) = {1901,1902};
Line(3) = {1902,1903};
Line(4) = {1903,1900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {1000};
Plane Surface(3) = {1,2};

Recombine Surface(3);

Physical Curve("top", 1) = {4};
Physical Curve("outflow", 2) = {1};
Physical Curve("bottom", 3) = {2};
Physical Curve("inflow", 4) = {3};
Physical Curve("aerofoil", 5) = {1000};
Physical Surface("interior", 6) = {3};
