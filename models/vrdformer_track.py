


class TrackingBase(nn.Module):
    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        
        
class VRDFormerTracking(TrackingBase, VRDFormer):
    def __init__(self, tracking_kwargs, detr_kwargs):
        VRDFormer.__init__(self, **detr_kwargs)
        TrackingBase.__init__(self, **tracking_kwargs)