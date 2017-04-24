import matplotlib.pyplot as plt

plt.plot([15,20,22,23,24,25,26,30,35,40,45,50,55], [2,1.9,1.9,1.7,1.2,1.9,1.9,2.2,2.1,2.0,2.3,2.3,2.2], 'ro')
plt.axis([10, 60, 1, 2.5])
plt.xlabel('threshold (X)')
plt.ylabel('error percentage (err)')
plt.show()

plt.plot([100,200,400,800,2000,4000],[14.04,9.81,5.37,3.86,2.8,1.9], 'ro')
plt.axis([0,4000,0,15])
plt.xlabel('training set (N)')
plt.ylabel('error percentage (err)')
plt.show()

plt.plot([100,200,400,800,2000,4000],[10,5,5,12,6,9], 'ro')
plt.axis([0,4000,0,15])
plt.xlabel('training set (N)')
plt.ylabel('iterations (iter)')
plt.show()