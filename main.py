from protargo import *

if __name__ == "__main__":
	ctx = DebateContext.get_instance()
	ctx.build(5)
	ctx.loop()
