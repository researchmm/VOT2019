from libs.core.evaluation import VOTDataset, TDataset, RGBTDataset, Tracker



# two nets
def res50_twonets(run_id):  # rgb of rgbt
    trackers = []
    for i in range(15):
        tracker_rgb = Tracker('improved', 'unrestore_res50_RGB', 'RGBandT', i, flag='RGB')
        tracker_t = Tracker('improved', 'unrestore_res50_T', 'RGBandT', i, flag='T')
        trackers.append([tracker_rgb, tracker_t])
    dataset = TDataset()

    return trackers, dataset


def res50_twonets_epoch(run_id):  # rgb of rgbt
    trackers = []
    for i in [14]:
        for j in range(1,5):
            tracker_rgb = Tracker('improved', 'unrestore_res50_RGB', 'RGBandT', j, i, flag='RGB')
            tracker_t = Tracker('improved', 'unrestore_res50_T', 'RGBandT', j, i, flag='T')
            trackers.append([tracker_rgb, tracker_t])
    dataset = TDataset()

    return trackers, dataset
