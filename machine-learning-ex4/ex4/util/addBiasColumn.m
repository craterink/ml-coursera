function mat = addBiasColumn(mat)

rows = size(mat, 1);
mat = [ones(rows, 1) mat];

