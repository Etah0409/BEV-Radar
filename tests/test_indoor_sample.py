import numpy as np

from mmdet3d.datasets.pipelines.indoor_sample import PointSample


def test_indoor_sample():
    np.random.seed(0)
    scannet_sample_points = PointSample(5)
    scannet_results = dict()
    scannet_points = np.array([[1.0719866, -0.7870435, 0.8408122, 0.9196809],
                               [1.103661, 0.81065744, 2.6616862, 2.7405548],
                               [1.0276475, 1.5061463, 2.6174362, 2.6963048],
                               [-0.9709588, 0.6750515, 0.93901765, 1.0178864],
                               [1.0578915, 1.1693821, 0.87503505, 0.95390373],
                               [0.05560996, -1.5688863, 1.2440368, 1.3229055],
                               [-0.15731563, -1.7735453, 2.7535574, 2.832426],
                               [1.1188195, -0.99211365, 2.5551798, 2.6340485],
                               [-0.9186557, -1.7041215, 2.0562649, 2.1351335],
                               [-1.0128691, -1.3394243, 0.040936, 0.1198047]])
    scannet_results['points'] = scannet_points
    scannet_instance_labels = np.array([15, 12, 11, 38, 0, 18, 17, 12, 17, 0])
    scannet_results['instance_labels'] = scannet_instance_labels
    scannet_pts_color = np.array([[77., 74., 63.], [176., 166., 156.],
                                  [178., 164., 153.], [240., 235., 237.],
                                  [58., 58., 59.], [245., 236., 229.],
                                  [158., 148., 141.], [195., 184., 178.],
                                  [193., 181., 174.], [105., 102., 97.]])
    scannet_results['pts_color'] = scannet_pts_color
    scannet_semantic_labels = np.array([38, 1, 1, 40, 0, 40, 1, 1, 1, 0])
    scannet_results['semantic_labels'] = scannet_semantic_labels
    scannet_results = scannet_sample_points(scannet_results, 0)
    scannet_point_cloud_result = scannet_results.get('points', None)
    scannet_pcl_color_result = scannet_results.get('pts_color', None)
    scannet_instance_labels_result = scannet_results.get(
        'instance_labels', None)
    scannet_semantic_labels_result = scannet_results.get(
        'semantic_labels', None)
    scannet_choices = np.array([2, 8, 4, 9, 1])
    assert np.allclose(scannet_points[scannet_choices],
                       scannet_point_cloud_result)
    assert np.allclose(scannet_pts_color[scannet_choices],
                       scannet_pcl_color_result)
    assert np.all(scannet_instance_labels[scannet_choices] ==
                  scannet_instance_labels_result)
    assert np.all(scannet_semantic_labels[scannet_choices] ==
                  scannet_semantic_labels_result)

    np.random.seed(0)
    sunrgbd_sample_points = PointSample(5)
    sunrgbd_results = dict()
    sunrgbd_point_cloud = np.array(
        [[-1.8135729e-01, 1.4695230e+00, -1.2780589e+00, 7.8938007e-03],
         [1.2581362e-03, 2.0561588e+00, -1.0341064e+00, 2.5184631e-01],
         [6.8236995e-01, 3.3611867e+00, -9.2599887e-01, 3.5995382e-01],
         [-2.9432583e-01, 1.8714852e+00, -9.0929651e-01, 3.7665617e-01],
         [-0.5024875, 1.8032674, -1.1403012, 0.14565146],
         [-0.520559, 1.6324949, -0.9896099, 0.2963428],
         [0.95929825, 2.9402404, -0.8746674, 0.41128528],
         [-0.74624217, 1.5244724, -0.8678476, 0.41810507],
         [0.56485355, 1.5747732, -0.804522, 0.4814307],
         [-0.0913099, 1.3673826, -1.2800645, 0.00588822]])
    sunrgbd_results['points'] = sunrgbd_point_cloud
    sunrgbd_results = sunrgbd_sample_points(sunrgbd_results, 0)
    sunrgbd_choices = np.array([2, 8, 4, 9, 1])
    sunrgbd_points_result = sunrgbd_results.get('points', None)
    assert np.allclose(sunrgbd_point_cloud[sunrgbd_choices],
                       sunrgbd_points_result)
