"""ACDC preprocessing contracts recovered from get_data_ready.ipynb."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None


class CropTests(unittest.TestCase):
    def test_crop_matches_notebook_bbox_and_edge_behavior(self):
        from scripts.preprocess_acdc import crop_by_mask
        image = np.arange(140 * 130).reshape(140, 130)
        label = np.zeros_like(image)
        label[60:80, 50:70] = 2
        crop = crop_by_mask(image, label, 96, label_value=2)
        np.testing.assert_array_equal(crop, image[21:117, 11:107])
        label[:] = 0
        label[:10, :10] = 2
        np.testing.assert_array_equal(crop_by_mask(image, label, 96, 2), image[:52, :52])
        with self.assertRaises(ValueError):
            crop_by_mask(image, np.zeros_like(label), 96, 2)

    def test_normalization_matches_numpy_notebook_and_opencv(self):
        from scripts.preprocess_acdc import normalize_crop
        image = np.array([[2., 88.], [174., 432.]])
        expected = image - image.min()
        expected = (expected / expected.max() * 255).astype(np.uint8)
        np.testing.assert_array_equal(normalize_crop(image, 'numpy'), expected)
        np.testing.assert_array_equal(normalize_crop(image, 'minmax_eps'),
            ((image-image.min())/(image.max()-image.min()+1e-8)*255).astype(np.uint8))
        np.testing.assert_array_equal(normalize_crop(image, 'opencv'),
                                     cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        np.testing.assert_array_equal(normalize_crop(np.ones((4,4)), 'numpy'), np.zeros((4,4),dtype=np.uint8))


@unittest.skipUnless(nib is not None, 'Install optional preprocessing dependencies')
class ACDCExportTests(unittest.TestCase):
    def test_all_time_frames_es_reference_fallback_and_flattened_names(self):
        from scripts.preprocess_acdc import export_patient
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patient = root / 'patient001'
            patient.mkdir()
            (patient / 'Info.cfg').write_text('ED: 1\nES: 2\n')
            x, y = np.indices((110,110))
            cine = np.empty((110,110,2,2))
            for z in range(2):
                for t in range(2): cine[:,:,z,t] = x*x+y*(t+1)+z*10
            label = np.zeros((110,110,2))
            label[50:60,50:60,0] = 2
            # Slice 1 has no mask and must use ES slice 0's location.
            nib.save(nib.Nifti1Image(cine,np.eye(4)), patient/'patient001_4d.nii.gz')
            nib.save(nib.Nifti1Image(label,np.eye(4)), patient/'patient001_frame02_gt.nii.gz')
            records = export_patient(patient, root/'out', reference_frame='ES',
                                     label_value=2, normalization='numpy', invalid_crop='error')
            self.assertEqual(len(records),4)
            self.assertEqual([r['file'] for r in records], [f'patient-1-frame{i}.png' for i in range(1,5)])
            self.assertTrue(records[1]['fallback_to_slice0'])
            self.assertFalse(records[0]['fallback_to_slice0'])
            crop=cine[6:102,6:102,1,0]
            expected=((crop-crop.min())/(crop.max()-crop.min())*255).astype(np.uint8)
            with Image.open(root/'out/patient-1-frame2.png') as image:
                np.testing.assert_array_equal(np.array(image),expected)
                self.assertEqual(image.mode,'L')
            selected = export_patient(patient, root/'filtered', reference_frame='ES',
                label_value=2, normalization='minmax_eps', empty_slices='skip', skip_constant=True)
            self.assertEqual(len(selected),2)
            self.assertEqual([r['file'] for r in selected],['patient-1-frame1.png','patient-1-frame2.png'])
            self.assertEqual([r['axial_slice'] for r in selected],[0,0])
