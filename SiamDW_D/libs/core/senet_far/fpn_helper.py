import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans

def pos_sz2bbox(pos, sz):
    cy, cx = pos
    h, w = sz
    lx = cx - w/2.
    ly = cy - h/2.
    return np.asarray([lx, ly, h, w])

def pos_sz2centerbox(pos, sz):
    cy, cx = pos
    h, w = sz
    return np.asarray([cx, cy, h, w])

def bbox2pos_sz(bbox):
    cx = bbox[0] + bbox[2]/2.
    cy = bbox[1] + bbox[3]/2.
    pos = [cy, cx]
    w, h = bbox[2:]
    sz = [h, w]
    return pos, sz

def l2_distance(vector_1, vector_2):
    # input is np.array
    return np.mean((vector_1 - vector_2)**2)

def cos_similarity(vector_1, vector_2):
    cos_sim = dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2))
    return cos_sim

def cal_simi(templates, fc_feature):
    simi_scores = []
    for template in templates:
        simi_scores.append(cos_similarity(template, fc_feature))
    return np.mean(simi_scores)

def get_box_hanning(bbox, history_box):
    R = 80.
    c_xy = bbox[:2] + 0.5*bbox[2:]
    c_hxy = history_box[:2] + 0.5*history_box[2:]
    bbox_offset = np.sum((c_xy - c_hxy)**2)
    r = np.sqrt(bbox_offset)
    r = min(r, R)
    coeff_window = (1. + np.cos(np.pi * r / R)) / 2.
    return coeff_window

def window_simi(bbox, new_fc_feature, history_box, history_fc_features):
    if len(history_box) == 0 or len(history_fc_features) == 0:
        coeff_window = 1.
        simi = 1.
    else:
        coeff_window = get_box_hanning(np.asarray(bbox), np.asarray(history_box))
        simi = cal_simi(history_fc_features, new_fc_feature)
    return coeff_window * simi

def find_envelope_box(proposals):
    # proposals lx ly w h
    bboxes = proposals.copy()
    lx = min(bboxes[:, 0])
    ly = min(bboxes[:, 1])
    rx = max(bboxes[:, 0] + bboxes[:, 2])
    ry = max(bboxes[:, 1] + bboxes[:, 3])
    w = rx - lx
    h = ry - ly
    return np.array([lx, ly, w, h])

def gen_adj_matrix(proposals):
    pos_vecs = proposals[:, :2] + 0.5*proposals[:, 2:]
    adj_matrix = np.zeros((len(pos_vecs), len(pos_vecs)))
    for i in range(len(pos_vecs)):
        for j in range(i):
            adj_matrix[i][j] = l2_distance(pos_vecs[i], pos_vecs[j])
            adj_matrix[j][i] = adj_matrix[i][j]
    return adj_matrix

def two_filter(iounet_scores, f_dists, topk=3):
    iou_indexes = np.argsort(iounet_scores)[::-1]
    d_indexes = np.argsort(f_dists)
    if len(d_indexes) < topk:
        topk = len(d_indexes)

    top_d_indexes = d_indexes[:topk]
    final_index = []
    for d_index in top_d_indexes:
        if iounet_scores[d_index]>0.3:
        # if d_index in top_iou_indexes:
            final_index.append(d_index)
    if len(final_index) == 0:
        final_index.append(top_d_indexes[0])
    return np.asarray(final_index)

def proposal_filter(proposals, scores, proposal_features, history_f, history_boxes):
    """
    proposals:
    scores: from iounet
    proposal_features: from iounet fc N*
    history_f: list of features of iounet fc
    history_boxes: latest boxes, may be usefull

    return: bbox which contain all left proposals
    TODO: add depth information to filter
    """
    history_len = history_f.shape[0]
    proposals_index = np.arange(len(proposals))

    # filter_num = 16
    filter_num = 24
    # stage one, filter acoording score
    sort_index = np.argsort(scores)[::-1]
    proposals = np.asarray(proposals)
    stage1_proposals = proposals[sort_index][:filter_num]
    stage1_proposal_features = proposal_features[sort_index][:filter_num]
    stage1_proposals_index = proposals_index[sort_index][:filter_num]

    # stage two, use kmeans to filter, and find best cluster, use iounet features
    # add history features to all features
    c_box = 2
    mix_proposals = stage1_proposals
    estimator_box = KMeans(n_clusters=c_box)
    f_sz = mix_proposals[:, 2] * mix_proposals[:, 3]
    f_sz = f_sz.reshape((-1, 1))
    f_pos = mix_proposals[:, :2] + 0.5*mix_proposals[:, 2:]
    f_stage1_proposals = np.concatenate((f_sz, f_pos), axis=1)
    estimator_box.fit(f_stage1_proposals)
    y_pred = estimator_box.labels_
    max_inds = np.argmax([sum(y_pred==i) for i in range(c_box)])
    stage2_proposals = stage1_proposals[y_pred == max_inds]
    stage2_proposal_features = stage1_proposal_features[y_pred == max_inds]
    stage2_proposals_index = stage1_proposals_index[y_pred == max_inds]
    # stage three, throw outler, use pos point Adjacent matrix
    if len(stage2_proposals) == 0:
        return find_envelope_box(stage1_proposals), stage1_proposals_index
    else:
        all_features = np.concatenate([history_f, stage2_proposal_features], axis=0)
        estimator = KMeans(n_clusters=2)
        estimator.fit(all_features)
        y_pred = estimator.labels_
        history_labels = y_pred[:history_len]
        history_label = np.round(history_labels[-1])
        proposal_labels = y_pred[history_len:]
        stage3_proposals = stage2_proposals[proposal_labels == history_label]
        stage3_proposals_index = stage2_proposals_index[proposal_labels == history_label]
        if len(stage3_proposals) == 0:
            return find_envelope_box(stage2_proposals), stage2_proposals_index
        envelope_box = find_envelope_box(stage3_proposals)
        return envelope_box, stage3_proposals_index

