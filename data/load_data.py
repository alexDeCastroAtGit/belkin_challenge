import os

"""
Using the unofficial Kaggle CLI.
"""

competition = u'belkin-energy-disaggregation-competition'
os.system(u'cd daAta/; kg download -u mineiro -p min3ir017 -c %s' % competition)