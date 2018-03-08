from __future__ import division

__author__ = "Alexis Coffer"
__copyright__ = "2018"
__credits__ = [Bio 393 - Compbio]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "__author__"
__email__ = "acoffer@uw.com"
__status__ = "Development"

from unittest import TestCase, main
from mass_extinctions import timeline, period #functions from project2

class geological_timelinesTests(TestCase):
    """ Tests for fitting timeline periods"""
    def setUp(self):
        #set up example data to fit timeline to age range
        data =[]
        self.data = data
        pass

    def test_period(self):
        """test period reports correct value on timeline age"""
        data = []
        obs = []
        print(obs)
        exp = [] #hand calculate this value
        self.assertEqual(obs,exp)
        # self.data = data basic format


if __name__ == "__main__":
    main()
