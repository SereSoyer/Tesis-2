import json
import unittest
import torch
from src.evaluation.classification import classification_metrics, ClassificationMetrics

class MetricsTests(unittest.TestCase):
    def test_known_example_and_probability_auc(self):
        m=classification_metrics([0,0,1,1],[.1,.4,.35,.8])
        self.assertEqual(m['confusion_matrix'],[[2,0],[1,1]])
        self.assertEqual(m['accuracy'],.75)
        self.assertEqual(m['precision'],1)
        self.assertEqual(m['recall'],.5)
        self.assertAlmostEqual(m['f1'],2/3)
        self.assertEqual(m['roc_auc'],.75)
        self.assertEqual(classification_metrics([0,1],[.5,.5])['confusion_matrix'],[[0,1],[0,1]])

    def test_single_class_and_json(self):
        m=classification_metrics([0,0],[.1,.2])
        self.assertIsNone(m['roc_auc'])
        self.assertEqual(m['precision'],0)
        self.assertEqual(set(m['undefined_metrics']),{'precision','recall','f1','roc_auc'})
        json.dumps(m,allow_nan=False)

    def test_unequal_batches_detach_and_reset(self):
        acc=ClassificationMetrics()
        acc.update(torch.tensor([0]),torch.tensor([.1],requires_grad=True))
        acc.update([0,1,1],[.4,.35,.8])
        self.assertEqual(acc.compute(),classification_metrics([0,0,1,1],[.1,.4,.35,.8]))
        acc.reset()
        with self.assertRaises(ValueError):acc.compute()

    def test_invalid_inputs(self):
        for y,p in [([],[]),([0],[.1,.2]),([2],[.5]),([0],[float('nan')]),([1],[2]),([[0,1]],[[.2,.3]])]:
            with self.subTest(y=y,p=p),self.assertRaises(ValueError):classification_metrics(y,p)
        with self.assertRaises(ValueError):classification_metrics([0],[.2],threshold=1.1)
