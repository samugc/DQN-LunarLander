import unittest
import numpy as np
from dqn.replay import SumTree, PrioritizedReplayMemory
class ReplayTests(unittest.TestCase):
    def test_priority_update_and_sampling_intervals(self):
        tree=SumTree(4)
        for p in [1.,2.,3.,4.]:tree.add(p)
        self.assertAlmostEqual(tree.total,10.)
        self.assertEqual([tree.get(x)[1] for x in [.5,2.,4.,8.]],[0,1,2,3])
        tree.update(3,5.);self.assertAlmostEqual(tree.total,14.)
    def test_ring_overwrites_leaves(self):
        tree=SumTree(2)
        for p in [1.,2.,7.]:tree.add(p)
        self.assertEqual(tree.n_entries,2);self.assertEqual(tree.total,9.)
    def test_non_power_of_two_capacity(self):
        tree=SumTree(5)
        for p in [1.,2.,3.,4.,5.]:tree.add(p)
        self.assertEqual(tree.total,15.)
        found={tree.get(x)[1] for x in np.linspace(.01,14.99,100)}
        self.assertEqual(found,set(range(5)))
    def test_empty_replay_rejected(self):
        with self.assertRaises(ValueError):PrioritizedReplayMemory(8).sample_memory(2,.4)
if __name__=='__main__':unittest.main()
