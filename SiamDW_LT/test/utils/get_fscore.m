function [persion, recall, fscore] = get_fscore(result_sequences, values, year)
%get_eao - Description
%
% Syntax: score = get_fscore(result_sequences, values, year)

% Input:
% - result_sequences (cell): A cell array of valid sequence descriptor structures.
% - values (cell): A cell array of confidence results

addpath('/data/home/v-had/data_local/matlab_gene/vot-toolkit/'); toolkit_path; % Make sure that VOT toolkit is in the path

pwd = ['/data/home/v-had/data_local/matlab_gene/vot-toolkit/', 'vot-workspace', year, 'lt']  % year is a str (can not be a number)
pwd()
[gt_sequences, experiments] = workspace_load(pwd);

experiment_sequences = 0}
(gt_sequences, experiments{1}.converter);

% from analyze_presion_recall.m
resolution = 100;
thresholds = [];
tags = {};

selectors = sequence_selectors(experiments, experiment_sequences);  % sequence_selectors from sequence/sequence_selectors.m
result.curves = cell(1, numel(selectors));
result.fmeasure = zeros(1, numel(selectors));
result.selectors = cellfun(@(x) x.name, selectors, 'UniformOutput', false);

thresholds = determine_thresholds(experiments, values, experiment_sequences, resolution);

for s = 1:numel(selectors)
    trajectories = cell(35, 1);
    trajectories{s} = result_sequences{s};
    values_temp = cell(35, 1);
    values_temp{s} = values{s};

    [curves, fmeasure] = calculate_tpr_fscore(selectors{s}, experiments, trajectories, values_temp, experiment_sequences, thresholds);

    result.curves{1, s} = curves;
    result.fmeasure(1, s) = fmeasure;

end;

% from report_presion_recall.m
average_curve = cell(1, 1);  % cell(numel(trackers), 1)
average_persion = zeros(1, 1);
average_recall = zeros(1, 1);
average_fmeasure = zeros(1, 1);
average_curve{1} = mean(cat(3, result.curves{1, :}), 3);
P_RETURN = average_curve{1}(:, 1);
R_RETURN = average_curve{1}(:, 2);
f = 2 * (P_RETURN .* R_RETURN) ./ (P_RETURN + R_RETURN);
[average_fmeasure(1), idx] = max(f);   % final f scores
average_persion(1) = P_RETURN(idx);   % final persion
average_recall(1) = R_RETURN(idx);   % final recall

persion = average_persion(1);
recall = average_recall(1);
fscore = average_fmeasure(1);

end

function [thresholds] = determine_thresholds(experiment, values, sequences, resolution)

    confidence_name = 'confidence';

    selector = sequence_tag_selectors(experiment, sequences, {'all'});
    certanty = zeros(sum(cellfun(@numel, values(:, 1), 'UniformOutput', true)), size(values, 2));

    i = 1;

    for s = 1:size(values, 1)

        for r = 1:size(values, 2)

            if isempty(values{s, r})
                continue;
            end;


            temp = cell2mat(values{s, r});
            certanty(i:i+size(values{s, r}')-1, r) = temp';  %zzp -modify

        end;
    end;

    thresholds = unique(certanty(~isnan(certanty)));

    if numel(thresholds) > resolution
        delta = floor(numel(thresholds) / (resolution - 2));
        idxs = round(linspace(delta, numel(thresholds)-delta, resolution-2));
        thresholds = thresholds(idxs);
    end

    thresholds = [-Inf; thresholds; Inf];
end


function [curve, fmeasure, fbest] = calculate_tpr_fscore(selector, experiment, trajectories, values, sequences, thresholds)
    % trajectories: 35*1 cell. only one has value, others is empty
    % values: 35*1 cell. only one has value, others is empty

    confidence_name = 'confidence';
    confidence_inverse = false;

    groundtruth = selector.groundtruth(sequences);

    overlaps = zeros(sum(cellfun(@numel, groundtruth, 'UniformOutput', true)), size(trajectories, 2));
    certanty = zeros(size(overlaps));

    i = 1;

    N = 0;

    for s = 1:numel(groundtruth)

        for r = 1:size(trajectories, 2)

            if isempty(trajectories{s, r})
                continue;
            end;


            tempT = trajectories{s, r}';   % zzp
            tempG = groundtruth{s};

            % zzp
            for id = 1:numel(tempT)
                tempT{id} = cell2mat(tempT{id});
            end


            [~, frames] = estimate_accuracy(tempT, tempG, ...
                'BindWithin', [sequences{s}.width, sequences{s}.height]);

            frames(isnan(frames)) = 0;

            overlaps(i:i+size(groundtruth{s})-1, r) = frames;


            temp = cell2mat(values{s, r});

            certanty(i:i+size(groundtruth{s})-1, r) = temp';  %zzp -modify

            % certanty(i:i+size(groundtruth{s})-1, r) = values{s, r};

        end;

        i = i + size(groundtruth{s});

        if ~isempty(groundtruth{s})
            N = N + sum(cellfun(@(x) numel(x) > 1, groundtruth{s}, 'UniformOutput', true));
        end;
    end;

    if isempty(thresholds)
       thresholds = certanty(~isnan(certanty));
    end

    thresholds = sort(thresholds, iff(confidence_inverse, 'descend', 'ascend'));

    curve = zeros(numel(thresholds), 3);

    curve(:, 3) = thresholds;

    for k = 1:numel(thresholds)

        % indicator vector where to calculate Pr-Re
        subset = certanty >= thresholds(k);

        if sum(subset) == 0
            % special case - no prediction is made:
            % Precision is 1 and Recall is 0
            curve(k,1) = 1;
            curve(k,2) = 0;
        else
            curve(k, 1) = mean(overlaps(subset));
            curve(k, 2) = sum(overlaps(subset)) ./ N;
        end

    end

    f = 2 * (curve(:, 1) .* curve(:, 2)) ./ (curve(:, 1) + curve(:, 2));

    [fmax, fidx] = max(f);

    fmeasure = fmax;
    fbest = thresholds(fidx);

 end



