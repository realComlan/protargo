class Argument:
  pass

class ArgumentGraph:
  pass

class UniversalGraph(ArgumentGraph):
  pass

class MergedGraph(ArgumentGraph):
  pass

class DebateSession:
  pass

class Debater:
  pass

class ExplainationStrategy:
  pass

class Explainer:
  def __init__(self, strategy=None):
      self.strategy = strategy
