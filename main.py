from protargo import *

if __name__ == "__main__":
	ArgumentGraph().generate().save("output1.apx")
	ArgumentGraph().generate().save("output2.apx")
	ArgumentGraph().generate().save("output3.apx")
	ArgumentGraph().generate().save("output4.apx")
	ArgumentGraph().generate().plot()
	ArgumentGraph().generate().plot()
