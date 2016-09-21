import numpy as np
import matplotlib.pyplot as plt

#valtest, testval, glsvm
x = [0,0.1,0.2,0.5,1.0];
y_100 = [0.774228924568995, 0.774228924568995, 0.774228924568995, 0.774228924568995, 0.774228924568995]
y_90 = [0.7842155848, 0.8007361645, 0.7627679508, 0.7649400236, 0.7762506531]
y_80 = [0.7741997877, 0.7663785186,0.7789112551,0.7746065017,0.7793458375]

#valval, testtest, glsvm
y_30=[0.577065765531, 0.671894423033, 0.654064310612, 0.607811655648, 0.555337760343,]
y_40=[0.70391744441, 0.722707449981, 0.719024475143, 0.683734798082, 0.693077957518,]
y_50=[0.749382094592, 0.761490319576, 0.757926987538, 0.7519690284, 0.719681862017,]
y_60=[0.746493053379, 0.766258823486, 0.762420296302, 0.741541565746, 0.678096715747,]
y_70=[0.767511903209, 0.76660856252, 0.760381986909, 0.757083881964, 0.741489140266,]
y_80=[0.783843088691, 0.791975508255, 0.791098793678, 0.779333073837, 0.77139870301,]
y_90=[0.734858400246, 0.751761726718, 0.739922391187, 0.747868432636, 0.747767119761,]
y_100=[0.716,0.716,0.716,0.716,0.716]

plt.figure(figsize=(8,6))

plt.xlabel("tradeoff")
plt.ylabel("map")
plt.title("map, scale=100")
plt.plot(x, y_100)
plt.show()