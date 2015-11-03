function xn = normalize01(x)

maxx = max(x);
minx = min(x);
xn = (x - minx)/(maxx - minx);