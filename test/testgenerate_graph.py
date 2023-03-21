from numpy import random
N=10
mat = [[0 for x in range(N)] for y in range(N)]
for _ in range(N):
	for j in range(5):
		v1 = random.randint(0,N-1)
		v2 = random.randint(0,N-1)
		if(v1 > v2):
			mat[v1][v2] = 1
		elif(v1 < v2):
			mat[v2][v1] = 1

for r in mat:
	print(','.join(map(str, r)))

