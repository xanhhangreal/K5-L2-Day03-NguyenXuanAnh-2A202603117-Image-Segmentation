import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import run


class MaskTests(unittest.TestCase):
    def test_cvat_rle_empty_full_and_disconnected(self):
        disconnected = np.zeros((7, 9), dtype=bool)
        disconnected[0, 0] = disconnected[6, 8] = True
        disconnected[2:4, 3:5] = True
        for mask in (np.zeros((7, 9), bool), np.ones((7, 9), bool), disconnected):
            points = run.encode(mask)
            self.assertEqual(points[-4:], [0, 0, 8, 6])
            np.testing.assert_array_equal(run.decode(points), mask)

    def test_invalid_rle_rejected(self):
        with self.assertRaises(AssertionError):
            run.decode([2, 3, 0, 0, 8, 6])

    def test_cvat_reordering_and_float_serialization(self):
        a = {'frame': 0, 'label_id': 1, 'type': 'mask', 'source': 'auto', 'points': [0, 4, 0, 0, 1, 1]}
        b = {**a, 'label_id': 2}
        returned = copy.deepcopy([b, a])
        for s in returned:
            s['points'] = list(map(float, s['points']))
        self.assertEqual(Counter(map(run.mask_signature, [a, b])), Counter(map(run.mask_signature, returned)))
        returned[0]['source'] = 'manual'
        self.assertNotEqual(Counter(map(run.mask_signature, [a, b])), Counter(map(run.mask_signature, returned)))

    def test_panoptic_precedence_and_semantic_taxonomy(self):
        class FakeModels:
            def detect(self, image, classes):
                return [{'label': 'car', 'confidence': .9, 'bbox': [1, 1, 3, 3]}]
            def sam(self, raw, box):
                mask = np.zeros((5, 5), bool)
                mask[1, 1] = mask[3, 3] = True
                return mask
            def segment(self, image):
                labels = np.zeros((5, 5), dtype=int)
                labels[0, :] = 1
                return labels, np.ones((5, 5)), {0: 'road', 1: 'unknown'}
        old_root = run.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run.ROOT = Path(tmp)
                image = Path(tmp) / 'test.jpg'
                Image.new('RGB', (5, 5)).save(image)
                result = run.predict(FakeModels(), image, {'type': 'panoptic'}, {'classes': ['road', 'car'], 'stuff': ['road']}, Path(tmp)/'predictions'/'test')
                masks = {m['label']: run.decode(m['points']) for m in result['masks']}
                self.assertFalse((masks['road'] & masks['car']).any())
                self.assertEqual(int(masks['car'].sum()), 2)
                self.assertEqual(result['unlabeled_pixels'], 5)
                self.assertEqual(set(masks), {'car', 'road'})
            finally:
                run.ROOT = old_root


if __name__ == '__main__':
    unittest.main()
