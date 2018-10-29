
class state(object):
	def __init__(self,theta,x,y):
		self.theta = theta
		self.x = x
		self.y = y

	def __mul__(self,other):
		return other*np.array([self.theta, self.x, self.y])