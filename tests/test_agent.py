import unittest
import numpy as np
import tensorflow as tf
from dqn.agent import DQN
class AgentTests(unittest.TestCase):
    def test_huber_weights_apply_to_each_sample(self):
        agent=DQN(8,np.int64(4))
        losses=agent.huber(tf.constant([[0.],[0.]]),tf.constant([[.5],[2.]]))
        self.assertEqual(tuple(losses.shape),(2,))
        np.testing.assert_allclose(losses.numpy(),[.125,1.5])
        self.assertAlmostEqual(float(tf.reduce_mean(losses*[1.,0.])),.0625)
    def test_synthetic_update_terminal_and_target_copy(self):
        tf.random.set_seed(7);agent=DQN(8,np.int64(4))
        states=tf.ones((4,8));actions=tf.constant([0,1,2,3])
        rewards=tf.constant([1.,-1.,2.,0.]);dones=tf.ones((4,))
        before=agent.model(states).numpy()[np.arange(4),actions.numpy()]
        errors=agent._train_step(states,actions,rewards,states,dones,tf.ones((4,)))
        np.testing.assert_allclose(errors.numpy(),rewards.numpy()-before,atol=1e-5)
        self.assertTrue(np.isfinite(errors.numpy()).all())
        agent.update_target_model(hard=True)
        for a,b in zip(agent.model.get_weights(),agent.target_model.get_weights()):np.testing.assert_array_equal(a,b)
    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path
        agent=DQN(8,np.int64(4));x=tf.ones((1,8));expected=agent.model(x).numpy()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'agent.keras';agent.model.save(p)
            loaded=tf.keras.models.load_model(p)
            np.testing.assert_allclose(loaded(x).numpy(),expected)
if __name__=='__main__':unittest.main()
