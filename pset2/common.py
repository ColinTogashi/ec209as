# TODO: script header

# create state class that has all properties of the state
class state(object):
    # initalize all properties
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading

    # change the equality so that any state with the same properties is equal
    def __eq__(self, another):
        return (hasattr(another, 'x') and self.x == another.x) \
            and (hasattr(another, 'y') and self.y == another.y) \
            and (hasattr(another, 'heading') and self.heading == another.heading)

    # redefine so that the state can be applied as a hash key
    def __hash__(self):
        return hash((self.x,self.y,self.heading))

    def __repr__(self):
        return 'state' + str((self.x, self.y, self.heading))


# create action class to describe all actions
class action(object):
    # initialize all properties
    def __init__(self, movement, rotation=None, vector=None):
        self.movement = movement
        self.rotation = rotation
        self.vector = vector

    # change the equality so that any action with the same properties is equal
    def __eq__(self, another):
        return (hasattr(another, 'movement') and self.movement == another.movement) \
            and (hasattr(another, 'rotation') and self.rotation == another.rotation) 

    # redefine so that the action can be applied as a hash key
    def __hash__(self):
        return hash((self.movement,self.rotation))

    def __repr__(self):
        return 'action' + str((self.movement, self.rotation))

if __name__ == '__main__':
    s = state(1,2,2)
    a = action('forwards', 'right')

    print s, a