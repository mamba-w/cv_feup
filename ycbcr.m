% Directory path where your images are stored
imageDir = 'Images/';
outputDir = 'Segmented/';
gtDir = 'Hand_masks/';

% Initialize variables to store TP, FP, and FN counts
TP = 0;
FP = 0;
FN = 0;

% Initialize variables to store bounding box data
allBoundingBoxes = {};

% Get a list of all image files in the directory
imageFiles = dir(fullfile(imageDir, '*.jpg'));

% Loop through each image file
for i = 1:numel(imageFiles)
    % Construct the full file path for the current image
    imageFilename = fullfile(imageDir, imageFiles(i).name);

    % Read the original image
    img = imread(imageFilename);

    % Step 1: Convert the image to the YCbCr color space
    ycbcrImg = rgb2ycbcr(img);

    % Step 2: Extract the Y, Cb, and Cr channels
    yChannel = ycbcrImg(:, :, 1);
    cbChannel = ycbcrImg(:, :, 2);
    crChannel = ycbcrImg(:, :, 3);

    % Step 3: Skin color segmentation based on Cb and Cr channels
    skinMask = (cbChannel >= 77 & cbChannel <= 127) & (crChannel >= 133 & crChannel <= 173);

    % Step 4: Refine segmentation using morphological operations
    skinMask = imfill(skinMask, 'holes');
    skinMask = bwareaopen(skinMask, 50); % Remove small noise regions

    % Remove smaller connected components
    skinMask = bwareafilt(skinMask, 1); % Keep only the largest connected component

    % Step 5: Use orthogonal projections to find arm and wrist boundaries
    
    % Calculate horizontal projection
    horizontalProjection = sum(skinMask, 2);
    
    % Smooth the horizontal projection to reduce noise
    smoothedHorizontalProjection = smoothdata(horizontalProjection, 'gaussian');
    
    % Find the significant valleys in the smoothed horizontal projection
    valleys = findvalleys(smoothedHorizontalProjection);
    
    % Estimate the wrist boundary based on the first significant valley if available
    if ~isempty(valleys)
        wristBoundary = valleys(1);
        % Remove the hand and wrist based on the estimated boundary
        skinMask(wristBoundary:end, :) = 0;
    end
    
    % Find bounding box of largest connected component in skin mask
    stats = regionprops('table', skinMask, 'BoundingBox');
    boundingBox = stats.BoundingBox;
    
    % Store bounding box data
    allBoundingBoxes{i} = boundingBox;
    
    % Draw bounding box on original image
    imgWithBoundingBox = insertShape(img, 'Rectangle', boundingBox, 'Color', 'red', 'LineWidth', 15);

    % Display the original image, skin mask, and image with bounding box
    figure('Name', ['Image ', num2str(i)]);
    
    subplot(1, 3, 1);
    imshow(img);
    title('Original Image');
    
    subplot(1, 3, 2);
    imshow(skinMask);
    title('Skin Mask (Hand and Wrist Removed)');
    
    subplot(1, 3, 3);
    imshow(imgWithBoundingBox);
    title('Original Image with Bounding Box');
    
    % Save the skin mask
    [~, imageName, ~] = fileparts(imageFiles(i).name);
    outputFilename = fullfile(outputDir, [imageName '_skin_mask_no_hand_wrist.png']);
    imwrite(skinMask, outputFilename);
end

% Evaluate performance metrics

% Loop through each image file
for i = 1:numel(imageFiles)
    % Load ground truth bounding box data
    gtFilename = fullfile(gtDir, [imageFiles(i).name(1:end-4) '_mask.png']);
    gtData = imread(gtFilename);
    gtBoundingBoxes = regionprops('table', gtData, 'BoundingBox');
    
    % Get detected bounding box for current image
    boundingBox = allBoundingBoxes{i};
    
    % Initialize logical array to track if each ground truth bounding box is detected
    gtDetected = false(size(gtBoundingBoxes, 1), 1);
    
    % Calculate Jaccard index for each detected bounding box
    for j = 1:size(boundingBox, 1)
        % Initialize maximum Jaccard index to 0
        maxJaccard = 0;
        
        % Check Jaccard index for each ground truth bounding box
        for k = 1:size(gtBoundingBoxes, 1)
            % Compute intersection and union areas
            intersectionArea = rectint(boundingBox(j,:), gtBoundingBoxes.BoundingBox(k,:));
            unionArea = rectint(boundingBox(j,:), boundingBox(j,:)) + rectint(gtBoundingBoxes.BoundingBox(k,:), gtBoundingBoxes.BoundingBox(k,:)) - intersectionArea;
            
            % Compute Jaccard index
            jaccardIndex = intersectionArea / unionArea;
            
            % Update maximum Jaccard index if higher
            maxJaccard = max(maxJaccard, jaccardIndex);
            
            % Mark ground truth bounding box as detected if Jaccard index is above threshold
            if jaccardIndex >= 0.5
                gtDetected(k) = true;
            end
        end
        
        % Check if maximum Jaccard index is equal to or larger than 0.5
        if maxJaccard >= 0.5
            TP = TP + 1; % True positive
        elseif maxJaccard > 0 && maxJaccard < 0.5
            FP = FP + 1; % False positive
        end
    end
    
    % Count undetected ground truth bounding boxes as false negatives
    FN = FN + sum(~gtDetected);
    
    % Clear variables for next iteration
    clear gtData gtBoundingBoxes gtDetected;
end

% Calculate recall, precision, and F1-measure
recall = TP / (TP + FN);
precision = TP / (TP + FP);
F1 = 2 * (precision * recall) / (precision + recall);

% Display evaluation results
disp(['True Positives (TP): ', num2str(TP)]);
disp(['False Positives (FP): ', num2str(FP)]);
disp(['False Negatives (FN): ', num2str(FN)]);
disp(['Recall (R): ', num2str(recall)]);
disp(['Precision (P): ', num2str(precision)]);
disp(['F1-measure (F1): ', num2str(F1)]);

% Function to find significant valleys in a signal
function valleys = findvalleys(signal)
    % Initialize variables
    valleys = [];
    isDescending = false;
    
    % Iterate through the signal
    for idx = 2:length(signal)
        % Check if the signal is descending
        if signal(idx) < signal(idx - 1)
            isDescending = true;
        elseif signal(idx) > signal(idx - 1) && isDescending
            % If the signal starts ascending after descending, mark a valley
            valleys = [valleys, idx - 1];
            isDescending = false;
        end
    end
end
