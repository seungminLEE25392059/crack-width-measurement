"""Lingyu Wu: shared measurement geometry, matching and deletion-risk regression tests."""
import unittest
import numpy as np
from src import measure,evaluate,method_d as d

class SharedMeasurementIntegrationTests(unittest.TestCase):
    def base(self):return np.zeros((64,64),bool)
    def check(self,mask,count):
        cracks,labels=measure.measure_cracks(mask)
        self.assertEqual(len(cracks),count);self.assertEqual(labels.shape,mask.shape)
        np.testing.assert_array_equal(labels>0,mask)
        self.assertEqual(sorted(row['id'] for row in cracks),list(range(1,count+1)))
        for row in cracks:
            self.assertTrue(set(measure.CRACK_KEYS)<=set(row))
            self.assertTrue(all(np.isfinite(row[k]) for k in ('length_px','mean_width_px','max_width_px')))
            self.assertGreater(row['mean_width_px'],0);self.assertGreaterEqual(row['max_width_px'],row['mean_width_px'])
        return cracks,labels
    def test_empty(self):self.check(self.base(),0)
    def test_single_pixel(self):
        x=self.base();x[20,20]=1;rows,_=self.check(x,1)
        self.assertEqual(rows[0]['max_width_px'],2);self.assertEqual(rows[0]['length_px'],0)
    def test_thin_line(self):
        x=self.base();x[20,5:35]=1;rows,_=self.check(x,1)
        self.assertEqual(rows[0]['mean_width_px'],2);self.assertEqual(rows[0]['length_px'],29)
    def test_wide_line(self):
        x=self.base();x[20:27,5:55]=1;rows,_=self.check(x,1)
        self.assertGreater(rows[0]['mean_width_px'],2);self.assertLessEqual(rows[0]['max_width_px'],8)
    def test_two_components(self):
        x=self.base();x[10,5:30]=1;x[40,5:30]=1;self.check(x,2)
    def test_branch(self):
        x=self.base();x[10:45,30]=1
        for k in range(15):x[10+k,30-k]=1;x[10+k,30+k]=1
        rows,_=self.check(x,1);self.assertGreater(rows[0]['length_px'],34)
    def test_diagonal_is_eight_connected(self):
        x=self.base();np.fill_diagonal(x[10:40,10:40],True);rows,_=self.check(x,1)
        self.assertAlmostEqual(rows[0]['length_px'],29*np.sqrt(2))
    def test_border(self):
        x=self.base();x[:5,:40]=1;rows,_=self.check(x,1);self.assertTrue(rows[0]['touches_border'])
    def test_severity_quantiles_and_boundaries(self):
        q=measure.severity_thresholds([2,3,4,5,6,7,8,9,10])
        np.testing.assert_allclose(q,np.quantile([2,3,4,5,6,7,8,9,10],[1/3,2/3]))
        self.assertEqual(measure.severity_grade(q[0],q),'moderate')
        self.assertEqual(measure.severity_grade(q[1],q),'severe')
        for values in ([],[0],[-1],[float('nan')],[float('inf')]):
            with self.subTest(values=values),self.assertRaises(ValueError):measure.severity_thresholds(values)
    def test_deleted_cracks_are_missed_not_zero_error(self):
        ref=self.base();ref[10,5:30]=1;ref[40,5:30]=1
        detail=d._measure_details(self.base(),ref)
        self.assertEqual(detail['missed'],2)
        self.assertTrue(all(row['status']=='missed' and 'mean_width_error' not in row for row in detail['errors']))
    def test_false_detection_and_fragment_counts(self):
        ref=self.base();ref[20,5:40]=1
        pred=ref.copy();pred[20,20:25]=0;pred[45,5:15]=1
        detail=d._measure_details(pred,ref)
        self.assertEqual(detail['missed'],0);self.assertEqual(detail['fragments'],1)
        self.assertEqual(detail['false_detections'],1)
    def test_shared_many_to_one_matching_is_exposed(self):
        ref=self.base();ref[10,5:30]=1;ref[40,5:30]=1
        pred=ref.copy();pred[10:41,10]=1
        detail=d._measure_details(pred,ref)
        self.assertEqual(detail['merged_predictions'],1);self.assertEqual(detail['merged_extra_references'],1)
        self.assertEqual(detail['missed'],0)

if __name__=='__main__':unittest.main(verbosity=2)
