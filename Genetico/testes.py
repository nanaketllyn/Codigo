import math

pixel = 0.5
a = 10000.0
m = 255.0
b = float((1.0 - math.exp(m/a))/m)

if pixel < 0.3:
    new_pixel = a * math.log10(1.0 + b * pixel)
    print new_pixel
else:
    new_pixel = (math.exp(pixel/a) - 1.0)/b
    print new_pixel
